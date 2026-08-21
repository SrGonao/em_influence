from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from em_influence.config import ExperimentManifest, load_manifest
from em_influence.jobs import build_jobs, discover_checkpoints, job_counts

ROOT = Path(__file__).parents[1]


def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(ROOT.parent.parent / "data/synthetic/train"))
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("BERGSON_BIN", "/bin/true")
    monkeypatch.setenv("MODEL_ARCHIVE_ROOT", "/mnt/cold-1/olmo_3_8b")


def test_cross_evaluation_manifest(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    manifest = load_manifest(ROOT / "experiments/cross_evaluation_olmo.yaml")
    jobs = build_jobs(manifest)
    counts = job_counts(jobs)
    ranked = [job for job in jobs if job.stage == "train" and job.parameters["query_suite"] != "random"]
    random = [job for job in jobs if job.stage == "train" and job.parameters["query_suite"] == "random"]
    assert len(ranked) == 270
    assert len(random) == 9
    suites = list(manifest.query_suites.values())
    flattened = [item for suite in suites for item in suite]
    question_ids = [item["id"] for item in yaml.safe_load(manifest.question_file.read_text())]
    assert len(flattened) == len(set(flattened)) == 44
    assert set(flattened) == set(question_ids)
    assert all(dataset.query_path.is_file() for dataset in manifest.datasets)
    assert all(dataset.checkpoint_path.is_dir() for dataset in manifest.datasets)
    assert counts == {"attribute": 9, "slice": 90, "train": 279, "evaluate": 837, "analyze": 1}


def test_filter_sweep_manifest(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    manifest = load_manifest(ROOT / "experiments/filter_sweep_career.yaml")
    jobs = build_jobs(manifest)
    counts = job_counts(jobs)
    # 1 dataset x 5 seeds baseline (train+evaluate), 4 methods x (2 modes x 5
    # fractions) slices, each slice trained+evaluated across all 5 seeds.
    assert counts == {"attribute": 4, "slice": 40, "train": 205, "evaluate": 205, "analyze": 1}
    attribution_jobs = [job for job in jobs if job.stage == "attribute"]
    assert len(attribution_jobs) == 4
    # Unlike cross_evaluation/training_time, a filter_sweep attribution job
    # depends on both the reference seed's judged completions (the query) and
    # its own trained model (what attribution actually ranks the data with).
    assert all(len(job.dependencies) == 2 for job in attribution_jobs)
    baseline_train = [job for job in jobs if job.stage == "train" and job.parameters["mode"] == "none"]
    assert len(baseline_train) == 5
    assert all(not job.dependencies for job in baseline_train)


def test_decile_sweep_manifest(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    manifest = load_manifest(ROOT / "experiments/decile_sweep_career.yaml")
    jobs = build_jobs(manifest)
    counts = job_counts(jobs)
    # 1 dataset x 5 seeds baseline (train+evaluate), 3 methods x 10 deciles,
    # each decile trained+evaluated across all 5 seeds.
    assert counts == {"attribute": 3, "slice": 30, "train": 155, "evaluate": 155, "analyze": 1}
    baseline_train = [job for job in jobs if job.stage == "train" and job.parameters.get("mode") == "none"]
    assert len(baseline_train) == 5

    # decile_sweep_career.yaml shares a results_root with filter_sweep_career.yaml
    # specifically so its baseline and (ekfac/wildguard/random) attribution jobs
    # are reused rather than recomputed - same {stage, parameters} means same id.
    sibling = load_manifest(ROOT / "experiments/filter_sweep_career.yaml")
    sibling_ids = {job.id for job in build_jobs(sibling)}
    assert {job.id for job in baseline_train} <= sibling_ids
    reused_methods = {"ekfac", "wildguard", "random"}
    attribution_ids = {job.id for job in jobs if job.stage == "attribute" and job.parameters["method"] in reused_methods}
    assert attribution_ids <= sibling_ids


def test_cross_model_sweep_manifest(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    manifest = load_manifest(ROOT / "experiments/cross_model_figure4_career.yaml")
    jobs = build_jobs(manifest)
    counts = job_counts(jobs)
    # 4 models x 5 seeds baseline (train+evaluate) + 4 attribution jobs;
    # 4 sources x 2 sides x 5 fractions = 40 slices; 1 target x 40 slices x 5
    # seeds = 200 cross-transfer train+evaluate jobs.
    assert counts == {"train": 220, "evaluate": 220, "attribute": 4, "slice": 40, "analyze": 1}
    cross_transfer = [job for job in jobs if job.stage == "train" and "target" in job.parameters]
    assert len(cross_transfer) == 200
    assert all(job.parameters["target"] == "olmo_3_7b" for job in cross_transfer)
    # A filtered dataset only depends on which model ranked it, so the same
    # slice is reused across every target that asks for it, not rebuilt.
    assert len({job.dependencies[0] for job in cross_transfer}) == 40

    # OLMo's baseline and cosine_similarity attribution are built identically
    # to filter_sweep_career.yaml's own baseline/attribution jobs, so sharing
    # a results_root reuses them instead of recomputing.
    sibling = load_manifest(ROOT / "experiments/filter_sweep_career.yaml")
    sibling_ids = {job.id for job in build_jobs(sibling)}
    olmo_baseline = {job.id for job in jobs if job.stage == "train" and job.parameters.get("mode") == "none"
                      and job.parameters.get("model") == "allenai/Olmo-3-7B-Instruct-SFT"}
    assert olmo_baseline <= sibling_ids
    olmo_attribution = {job.id for job in jobs if job.stage == "attribute"
                         and job.parameters.get("model") == "allenai/Olmo-3-7B-Instruct-SFT"}
    assert olmo_attribution <= sibling_ids


def _base_manifest_kwargs(tmp_path):
    template = tmp_path / "template.json"
    template.write_text("{}")
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt":"p","completion":"c"}\n')
    questions = tmp_path / "questions.yaml"
    questions.write_text("[]\n")
    return {
        "version": 1, "name": "rubric_validation", "kind": "filter_sweep",
        "results_root": tmp_path / "results",
        "model": {"model_id": "model", "training_template": template},
        "datasets": [{"name": "data", "path": data}],
        "question_file": questions,
        "query_suites": {"full": ["q"]}, "evaluation_suites": {"full": ["q"]},
        "training_seeds": [0],
        "attribution": {"methods": ["rubric"], "bergson_bin": "/bin/bergson"},
        "filter": {"selection_mode": "remove", "fractions": [0.1]},
        "resources": {"cuda_devices": [0]},
    }


def test_rubric_method_requires_rubric_block(tmp_path):
    with pytest.raises(ValidationError, match="rubric"):
        ExperimentManifest.model_validate(_base_manifest_kwargs(tmp_path))


def test_rubric_method_rejected_outside_filter_sweep(tmp_path):
    kwargs = _base_manifest_kwargs(tmp_path)
    kwargs["kind"] = "decile_sweep"
    kwargs["slicing"] = {"mode": "deciles"}
    kwargs["filter"] = None
    kwargs["rubric"] = {"metrics": ["wrongness"]}
    with pytest.raises(ValidationError, match="filter_sweep"):
        ExperimentManifest.model_validate(kwargs)


def test_filter_sweep_rubric_manifest(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    manifest = load_manifest(ROOT / "experiments/filter_sweep_career_rubric.yaml")
    jobs = build_jobs(manifest)
    counts = job_counts(jobs)
    # 1 dataset x 5 seeds baseline (train+evaluate); 5 rubric metrics, each its
    # own attribution job x (2 modes x 5 fractions) slices, each trained+
    # evaluated across all 5 seeds.
    assert counts == {"attribute": 5, "slice": 50, "train": 255, "evaluate": 255, "analyze": 1}
    attribution_jobs = [job for job in jobs if job.stage == "attribute"]
    assert {job.parameters["metric"] for job in attribution_jobs} == {
        "wrongness", "harm_potential", "overconfidence", "vulnerability", "subtlety",
    }
    assert len({job.id for job in attribution_jobs}) == 5  # distinct metrics hash to distinct jobs
    assert all(job.parameters["judge_model"] == "openai/gpt-5.4-nano" for job in attribution_jobs)
    assert all(len(job.dependencies) == 2 for job in attribution_jobs)

    # The baseline train run is built identically to filter_sweep_career.yaml's
    # own baseline job, so sharing a results_root reuses it instead of
    # recomputing - same construction as decile_sweep/cross_model_sweep.
    sibling = load_manifest(ROOT / "experiments/filter_sweep_career.yaml")
    sibling_ids = {job.id for job in build_jobs(sibling)}
    baseline_train = [job for job in jobs if job.stage == "train" and job.parameters.get("mode") == "none"]
    assert len(baseline_train) == 5
    assert {job.id for job in baseline_train} <= sibling_ids

    assert manifest.rubric.scores_root.is_dir()


def test_cross_evaluation_career_query_set_manifest(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    manifest = load_manifest(ROOT / "experiments/cross_evaluation_career.yaml")
    jobs = build_jobs(manifest)
    counts = job_counts(jobs)
    # 3 query suites x 10 deciles slices, each trained x 3 seeds (+3 random
    # baselines) = 93 train jobs, each evaluated against all 3 evaluation
    # suites = 279 evaluate jobs.
    assert counts == {"attribute": 3, "slice": 30, "train": 93, "evaluate": 279, "analyze": 1}

    # This manifest reproduces Figures A3/A4 without training its own
    # baseline: dataset.checkpoint_path/query_path are hardcoded paths into
    # filter_sweep_career.yaml's own baseline train/evaluate artifacts. Guard
    # against that hardcoding silently drifting out of sync if the sibling
    # manifest's model/dataset/seed-0 ever change.
    sibling = load_manifest(ROOT / "experiments/filter_sweep_career.yaml")
    sibling_jobs = build_jobs(sibling)
    reference_seed = sibling.training_seeds[0]
    baseline_train = next(
        job for job in sibling_jobs if job.stage == "train"
        and job.parameters.get("mode") == "none" and job.parameters["seed"] == reference_seed
    )
    baseline_evaluate = next(
        job for job in sibling_jobs if job.stage == "evaluate" and job.dependencies == (baseline_train.id,)
    )
    dataset = manifest.datasets[0]
    assert dataset.checkpoint_path == sibling.results_root / "artifacts" / baseline_train.id / "model"
    assert dataset.query_path == sibling.results_root / "artifacts" / baseline_evaluate.id / "answers.csv"


def test_training_time_manifest(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    manifest = load_manifest(ROOT / "experiments/training_time_olmo_auto.yaml")
    jobs = build_jobs(manifest)
    causal = [job for job in jobs if job.stage == "train"]
    assert len(causal) == 75
    observations = [job for job in jobs if job.stage == "evaluate" and job.parameters.get("phase") == "observational"]
    assert len(observations) == 38
    assert discover_checkpoints(manifest.existing_artifacts.checkpoint_root) == list(range(10, 361, 10)) + [369]
    assert manifest.checkpoints == [10, 20, 50, 100, 200, 369]
    assert manifest.existing_artifacts.fixed_attribution_root.is_dir()
