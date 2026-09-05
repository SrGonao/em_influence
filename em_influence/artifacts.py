from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactMetadata:
    job_id: str
    input_fingerprint: str
    status: str
    command: list[str] | None
    completed_at: str | None
    parameters: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    output_fingerprint: str | None = None


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def metadata_path(output: Path) -> Path:
    return output / ".em_influence.json"


def read_metadata(output: Path) -> ArtifactMetadata | None:
    path = metadata_path(output)
    if not path.is_file():
        return None
    try:
        return ArtifactMetadata(**json.loads(path.read_text()))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def is_complete(output: Path, job_id: str, input_fingerprint: str) -> bool:
    metadata = read_metadata(output)
    return bool(
        metadata
        and metadata.status == "complete"
        and metadata.job_id == job_id
        and metadata.input_fingerprint == input_fingerprint
    )


def write_metadata(
    output: Path,
    *,
    job_id: str,
    input_fingerprint: str,
    status: str,
    command: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
    configuration: dict[str, Any] | None = None,
    output_fingerprint: str | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    completed_at = (
        datetime.now(timezone.utc).isoformat() if status == "complete" else None
    )
    metadata = ArtifactMetadata(
        job_id=job_id,
        input_fingerprint=input_fingerprint,
        status=status,
        command=command,
        completed_at=completed_at,
        parameters=parameters,
        configuration=configuration,
        output_fingerprint=output_fingerprint,
    )
    destination = metadata_path(output)
    fd, temporary = tempfile.mkstemp(prefix=destination.name, dir=output)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(asdict(metadata), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def iter_stage_artifacts(results_root: Path, stage: str) -> list[tuple[ArtifactMetadata, Path]]:
    """Every completed job of one stage under a results_root, as
    (metadata, artifact_dir) pairs - metadata.parameters carries the job's
    identity (dataset, model, method, ...) and artifact_dir is where its
    output files (answers.csv, attributions.csv, ...) live. Generalizes the
    `evaluate`-only scan `write_run_manifest` does, for analysis code that
    needs `attribute` (or any other stage's) artifacts instead."""
    results: list[tuple[ArtifactMetadata, Path]] = []
    for meta_path in sorted((results_root / "artifacts").glob("*/.em_influence.json")):
        metadata = read_metadata(meta_path.parent)
        if metadata is None or metadata.status != "complete" or not metadata.job_id.startswith(f"{stage}__"):
            continue
        results.append((metadata, meta_path.parent))
    return results


def write_run_manifest(results_root: Path) -> Path:
    """Collect every completed evaluate job's parameters + judged answers.csv
    into one flat CSV, so plotting never has to parse a run's identity back
    out of a directory name - it's already sitting in job.parameters."""
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    for metadata, directory in iter_stage_artifacts(results_root, "evaluate"):
        answers_csv = directory / "answers.csv"
        if not answers_csv.is_file():
            continue
        row = {**(metadata.parameters or {}), "answers_csv": str(answers_csv)}
        for key in row:
            if key not in fields:
                fields.append(key)
        rows.append(row)
    path = results_root / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path

