"""One inexpensive check that all paper templates can be planned offline."""
from pathlib import Path

from em_influence import config
from em_influence.adapters import commands_for_job
from em_influence.config import load_manifest
from em_influence.jobs import build_jobs
from em_influence.runner import _layers

ROOT = Path(__file__).parents[1]


def test_all_paper_templates_plan_without_hardware_or_artifacts(tmp_path, monkeypatch):
    for variable, value in {"DATA_ROOT": "data", "RESULTS_ROOT": "results",
                            "MODEL_ARCHIVE_ROOT": "models", "BERGSON_BIN": "bergson"}.items():
        monkeypatch.setenv(variable, str(tmp_path / value))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def no_detection():
        raise AssertionError("Planning must not probe hardware")

    monkeypatch.setattr(config, "_detect_cuda_device_count", no_detection)
    for recipe in sorted((ROOT / "experiments").rglob("*.yaml")):
        manifest = load_manifest(recipe)
        jobs = build_jobs(manifest)
        seen = set()
        for layer in _layers(jobs):
            for job in layer:
                assert set(job.dependencies) <= seen, str(recipe)
                commands = commands_for_job(manifest, job, ROOT)
                assert commands or job.parameters.get("query_mode") == "fixed_final_query", str(recipe)
            seen.update(job.id for job in layer)
    assert not list(tmp_path.iterdir()), "Planning wrote files"
