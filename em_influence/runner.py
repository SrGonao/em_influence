from __future__ import annotations

from pathlib import Path

import yaml

from .adapters import artifact_dir, commands_for_job
from .artifacts import fingerprint, is_complete, read_metadata, write_metadata, write_run_manifest
from .config import ExperimentManifest
from .executor import LocalGpuExecutor
from .jobs import Job


def _layers(jobs: list[Job]) -> list[list[Job]]:
    """Group jobs into dependency-respecting layers (Kahn's algorithm): every
    job in a layer has all its in-graph dependencies satisfied by an earlier
    layer, so a layer's jobs can all run concurrently."""
    ids = {job.id for job in jobs}
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
    return bool(metadata and metadata.status == "complete")


def run_jobs(manifest: ExperimentManifest, jobs: list[Job], *, resume: bool, repo: Path) -> int:
    manifest.results_root.mkdir(parents=True, exist_ok=True)
    resolved = manifest.model_dump(mode="json")
    (manifest.results_root / "resolved_manifest.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    selected = {job.id for job in jobs}
    failed: set[str] = set()
    executor = LocalGpuExecutor(manifest.resources.cuda_devices, gpus_per_job=manifest.resources.gpus_per_job, jobs_per_gpu_group=manifest.resources.jobs_per_gpu_group)

    for layer in _layers(jobs):
        batch: list[tuple[Job, Path, str, list]] = []
        for job in layer:
            if any(dependency in failed for dependency in job.dependencies):
                failed.add(job.id)
                continue
            missing_dependencies = [dependency for dependency in job.dependencies if dependency not in selected and not _artifact_complete(manifest, dependency)]
            if missing_dependencies:
                raise RuntimeError(f"Job {job.id} requires incomplete stage dependencies: {missing_dependencies}")
            output = artifact_dir(manifest, job)
            digest = fingerprint({"manifest": resolved, "job": job.as_dict()})
            if resume and is_complete(output, job.id, digest):
                continue
            commands = commands_for_job(manifest, job, repo)
            write_metadata(output, job_id=job.id, input_fingerprint=digest, status="running",
                            command=[arg for command in commands for arg in command.argv],
                            parameters=job.parameters, configuration=resolved)
            batch.append((job, output, digest, commands))

        if not batch:
            continue
        chain_results = executor.run_chains([commands for _, _, _, commands in batch])
        for (job, output, digest, commands), results in zip(batch, chain_results):
            command_argv = [arg for command in commands for arg in command.argv]
            status = "failed" if any(result.returncode for result in results) else "complete"
            write_metadata(output, job_id=job.id, input_fingerprint=digest, status=status,
                            command=command_argv, parameters=job.parameters, configuration=resolved)
            if status == "failed":
                failed.add(job.id)

    write_run_manifest(manifest.results_root)
    return 1 if failed else 0
