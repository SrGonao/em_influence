from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import yaml

from .adapters import artifact_dir, commands_for_job
from .artifacts import is_complete, read_metadata, write_metadata, write_run_manifest
from .config import ExperimentManifest
from .executor import LocalGpuExecutor
from .jobs import Job
from .provenance import job_fingerprint, output_digest


def _layers(jobs: list[Job]) -> list[list[Job]]:
    """Group jobs into dependency-respecting layers (Kahn's algorithm): every
    job in a layer has all its in-graph dependencies satisfied by an earlier
    layer, so a layer's jobs can all run concurrently."""
    ids = {job.id for job in jobs}
    if len(ids) != len(jobs):
        raise ValueError("Job graph contains duplicate artifact IDs")
    done: set[str] = set()
    remaining = list(jobs)
    layers: list[list[Job]] = []
    while remaining:
        ready = [job for job in remaining if all(dependency not in ids or dependency in done for dependency in job.dependencies)]
        if not ready:
            raise RuntimeError("Job graph has a cycle or an unresolved dependency")
        layers.append(ready)
        done.update(job.id for job in ready)
        ready_ids = {job.id for job in ready}
        remaining = [job for job in remaining if job.id not in ready_ids]
    return layers


def _artifact_complete(manifest: ExperimentManifest, job_id: str) -> bool:
    metadata = read_metadata(manifest.results_root / "artifacts" / job_id)
    if metadata is None or metadata.status != "complete" or metadata.output_fingerprint is None:
        return False
    job = Job(job_id.split("__", 1)[0], metadata.parameters or {})
    # The stored directory is authoritative for an external dependency.
    if job.id != job_id:
        return False
    return output_digest(manifest, job) == metadata.output_fingerprint


def run_jobs(manifest: ExperimentManifest, jobs: list[Job], *, resume: bool, repo: Path) -> int:
    manifest = manifest.model_copy(update={"resources": manifest.resources.resolved()})
    manifest.results_root.mkdir(parents=True, exist_ok=True)
    resolved = manifest.model_dump(mode="json")
    (manifest.results_root / "resolved_manifest.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    selected = {job.id for job in jobs}
    failed: set[str] = set()
    executor = LocalGpuExecutor(manifest.resources.cuda_devices, jobs_per_gpu_group=manifest.resources.jobs_per_gpu_group)

    for layer in _layers(jobs):
        batch: list[tuple[Job, Path, str, list]] = []
        for job in layer:
            if any(dependency in failed for dependency in job.dependencies):
                failed.add(job.id)
                write_metadata(artifact_dir(manifest, job), job_id=job.id,
                               input_fingerprint="", status="blocked",
                               parameters=job.parameters, configuration=resolved)
                continue
            missing_dependencies = [dependency for dependency in job.dependencies if dependency not in selected and not _artifact_complete(manifest, dependency)]
            if missing_dependencies:
                raise RuntimeError(f"Job {job.id} requires incomplete stage dependencies: {missing_dependencies}")
            output = artifact_dir(manifest, job)
            commands = commands_for_job(manifest, job, repo)
            command_argv = [arg for command in commands for arg in command.argv]
            digest = job_fingerprint(manifest, job, commands)
            metadata = read_metadata(output)
            if (resume and is_complete(output, job.id, digest)
                    and metadata.output_fingerprint is not None
                    and output_digest(manifest, job) == metadata.output_fingerprint):
                continue
            # Driver scripts may skip existing outputs. Keep the old artifact
            # for inspection, but execute invalidated jobs in a clean directory.
            if output.exists():
                previous = manifest.results_root / ".previous" / f"{job.id}__{uuid4().hex}"
                previous.parent.mkdir(parents=True, exist_ok=True)
                output.rename(previous)
            write_metadata(output, job_id=job.id, input_fingerprint=digest, status="running",
                            command=command_argv, parameters=job.parameters, configuration=resolved)
            batch.append((job, output, digest, commands))

        if not batch:
            continue
        chain_results = executor.run_chains([commands for _, _, _, commands in batch])
        for (job, output, digest, commands), results in zip(batch, chain_results):
            command_argv = [arg for command in commands for arg in command.argv]
            produced = output_digest(manifest, job)
            status = "failed" if any(result.returncode for result in results) or produced is None else "complete"
            write_metadata(output, job_id=job.id, input_fingerprint=digest, status=status,
                            command=command_argv, parameters=job.parameters, configuration=resolved,
                            output_fingerprint=produced if status == "complete" else None)
            if status == "failed":
                failed.add(job.id)

    write_run_manifest(manifest.results_root)
    return 1 if failed else 0
