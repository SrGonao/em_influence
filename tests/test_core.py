import json
import sys
from pathlib import Path

import numpy as np

from em_influence.adapters import commands_for_job
from em_influence.artifacts import fingerprint, is_complete, iter_stage_artifacts, write_metadata, write_run_manifest
from em_influence.config import ExperimentManifest
from em_influence.jobs import Job
from em_influence.compat import filter_csv, filter_questions, slice_dataset, training_config
from em_influence.executor import Command, LocalGpuExecutor
from em_influence.selection import complement, deciles, extreme, random_subset, resample


def test_selection_is_deterministic_and_disjoint():
    scores = np.arange(100, dtype=float)
    slices = deciles(scores)
    joined = np.concatenate([item.indices for item in slices])
    assert len(set(joined)) == 100
    assert slices[0].indices[0] == 99
    assert set(extreme(scores, fraction=.1, side="top").indices) == set(range(90, 100))
    assert np.array_equal(random_subset(100, fraction=.1, seed=4).indices, random_subset(100, fraction=.1, seed=4).indices)
    assert len(complement(100, np.arange(10))) == 90
    assert len(resample(np.arange(10), target_size=100, seed=1)) == 100


def test_artifact_compatibility(tmp_path):
    digest = fingerprint({"a": 1})
    write_metadata(tmp_path, job_id="job", input_fingerprint=digest, status="complete", command=["true"])
    assert is_complete(tmp_path, "job", digest)
    assert not is_complete(tmp_path, "job", fingerprint({"a": 2}))


def test_write_run_manifest_collects_only_complete_evaluate_jobs(tmp_path):
    results_root = tmp_path / "results"
    artifacts = results_root / "artifacts"

    complete_eval = artifacts / "evaluate__career__abc123"
    (complete_eval).mkdir(parents=True)
    (complete_eval / "answers.csv").write_text("question_id,question,answer,aligned\na,q,a,5\n")
    write_metadata(complete_eval, job_id="evaluate__career__abc123", input_fingerprint="x", status="complete",
                    parameters={"dataset": "career", "method": "ekfac", "mode": "remove_top", "fraction": 0.1, "seed": 0})

    failed_eval = artifacts / "evaluate__career__def456"
    failed_eval.mkdir(parents=True)
    write_metadata(failed_eval, job_id="evaluate__career__def456", input_fingerprint="x", status="failed",
                    parameters={"dataset": "career", "method": "ekfac", "mode": "remove_bottom", "fraction": 0.1, "seed": 0})

    non_evaluate = artifacts / "train__career__ghi789"
    non_evaluate.mkdir(parents=True)
    write_metadata(non_evaluate, job_id="train__career__ghi789", input_fingerprint="x", status="complete",
                    parameters={"dataset": "career", "seed": 0})

    path = write_run_manifest(results_root)
    rows = path.read_text().splitlines()
    assert len(rows) == 2  # header + the one complete evaluate job
    assert "career" in rows[1] and "ekfac" in rows[1] and "remove_top" in rows[1]


def test_iter_stage_artifacts_filters_by_stage_and_status(tmp_path):
    results_root = tmp_path / "results"
    artifacts = results_root / "artifacts"

    complete_attr = artifacts / "attribute__career__qwen2.5_7b__abc"
    complete_attr.mkdir(parents=True)
    write_metadata(complete_attr, job_id="attribute__career__qwen2.5_7b__abc", input_fingerprint="x",
                    status="complete", parameters={"dataset": "career", "model": "qwen2.5_7b", "method": "cosine_similarity"})

    failed_attr = artifacts / "attribute__career__qwen3_8b__def"
    failed_attr.mkdir(parents=True)
    write_metadata(failed_attr, job_id="attribute__career__qwen3_8b__def", input_fingerprint="x", status="failed")

    an_evaluate = artifacts / "evaluate__career__ghi"
    an_evaluate.mkdir(parents=True)
    write_metadata(an_evaluate, job_id="evaluate__career__ghi", input_fingerprint="x", status="complete")

    results = iter_stage_artifacts(results_root, "attribute")
    assert len(results) == 1
    metadata, artifact_dir = results[0]
    assert metadata.parameters == {"dataset": "career", "model": "qwen2.5_7b", "method": "cosine_similarity"}
    assert artifact_dir == complete_attr


def test_executor_logs_and_failure(tmp_path):
    executor = LocalGpuExecutor([2])
    commands = [
        Command("ok", (sys.executable, "-c", "print('ok')"), tmp_path),
        Command("bad", (sys.executable, "-c", "raise SystemExit(7)"), tmp_path),
    ]
    results = {item.id: item for item in executor.run(commands)}
    assert results["ok"].returncode == 0
    assert results["bad"].returncode == 7
    assert results["ok"].stdout_path.read_text().strip() == "ok"


def test_executor_chains_run_in_order_and_stop_on_failure(tmp_path):
    executor = LocalGpuExecutor([0, 1], jobs_per_gpu_group=1)
    good_chain = [
        Command("good-a", (sys.executable, "-c", "print('a')"), tmp_path),
        Command("good-b", (sys.executable, "-c", "print('b')"), tmp_path),
    ]
    failing_chain = [
        Command("bad-a", (sys.executable, "-c", "raise SystemExit(3)"), tmp_path),
        Command("bad-b", (sys.executable, "-c", "print('never runs')"), tmp_path),
    ]
    results = executor.run_chains([good_chain, failing_chain])
    assert [result.id for result in results[0]] == ["good-a", "good-b"]
    assert all(result.returncode == 0 for result in results[0])
    # The chain stops at its first failure - "bad-b" never gets a result.
    assert [result.id for result in results[1]] == ["bad-a"]
    assert results[1][0].returncode == 3



def _manifest(tmp_path, method):
    template = tmp_path / "template.json"
    template.write_text("{}")
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt":"p","completion":"c"}\n')
    questions = tmp_path / "questions.yaml"
    questions.write_text("[]\n")
    return ExperimentManifest.model_validate({
        "version": 1, "name": "adapter", "kind": "cross_evaluation",
        "results_root": tmp_path / "results",
        "model": {"model_id": "model", "training_template": template},
        "datasets": [{"name": "data", "path": data, "query_path": tmp_path / "query.csv"}],
        "question_file": questions,
        "query_suites": {"full": ["q"]}, "evaluation_suites": {"full": ["q"]},
        "training_seeds": [0],
        "attribution": {"methods": [method], "bergson_bin": "/bin/bergson"},
        "slicing": {"mode": "deciles"},
        "resources": {"cuda_devices": [0]},
    })


def test_cosine_adapter_uses_build_and_score(tmp_path):
    manifest = _manifest(tmp_path, "cosine_similarity")
    commands = commands_for_job(manifest, Job("attribute", {"dataset": "data", "query_suite": "full"}), tmp_path)
    assert len(commands) == 4
    assert [command.argv[1] for command in commands[1:3]] == ["build", "score"]
    assert commands[0].argv[3] == "filter-csv"
    assert commands[3].argv[2] == "em_influence.bergson_export"


def test_length_adapter_uses_manifest_model_as_tokenizer(tmp_path):
    manifest = _manifest(tmp_path, "length")
    commands = commands_for_job(manifest, Job("attribute", {"dataset": "data", "method": "length"}), tmp_path)
    assert len(commands) == 1
    assert commands[0].argv[1].endswith("compute_length_attribution.py")
    assert commands[0].argv[-1] == "model"  # falls back to manifest.model.model_id (no training dependency)


def test_loss_adapter_uses_dependency_checkpoint(tmp_path):
    manifest = _manifest(tmp_path, "loss")
    job = Job("attribute", {"dataset": "data", "method": "loss"}, dependencies=("evalX", "trainX"))
    commands = commands_for_job(manifest, job, tmp_path)
    assert len(commands) == 1
    assert commands[0].argv[1].endswith("compute_loss_attribution.py")
    assert commands[0].argv[-1] == str(manifest.results_root / "artifacts" / "trainX" / "model")


def _rubric_manifest(tmp_path, scores_root=None, backend="openrouter", judge_model="openai/gpt-5.4-nano",
                     tensor_parallel_size=1, execution=None):
    template = tmp_path / "template.json"
    template.write_text("{}")
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt":"p","completion":"c"}\n')
    questions = tmp_path / "questions.yaml"
    questions.write_text("[]\n")
    return ExperimentManifest.model_validate({
        "version": 1, "name": "rubric_adapter", "kind": "filter_sweep",
        "results_root": tmp_path / "results",
        "model": {"model_id": "model", "training_template": template},
        "datasets": [{"name": "data", "path": data}],
        "question_file": questions,
        "query_suites": {"full": ["q"]}, "evaluation_suites": {"full": ["q"]},
        "training_seeds": [0],
        "attribution": {"methods": ["rubric"], "bergson_bin": "/bin/bergson"},
        "rubric": {"judge_model": judge_model, "metrics": ["wrongness"], "scores_root": scores_root,
                   "backend": backend, "tensor_parallel_size": tensor_parallel_size},
        "filter": {"selection_mode": "remove", "fractions": [0.1]},
        "resources": {"cuda_devices": [0]},
        **({"execution": execution} if execution else {}),
    })


def test_rubric_adapter_reuses_scores_file_when_present(tmp_path):
    scores_root = tmp_path / "scores"
    scores_root.mkdir()
    (scores_root / "data__openai_gpt-5.4-nano.jsonl").write_text('{"item_id": "data:1", "wrongness": 7}\n')
    manifest = _rubric_manifest(tmp_path, scores_root=scores_root)
    job = Job("attribute", {"dataset": "data", "method": "rubric", "metric": "wrongness", "judge_model": "openai/gpt-5.4-nano"},
              dependencies=("evalX", "trainX"))
    commands = commands_for_job(manifest, job, tmp_path)
    assert len(commands) == 1
    assert commands[0].argv[1].endswith("compute_rubric_attribution.py")
    assert "--scores-file" in commands[0].argv
    assert commands[0].argv[commands[0].argv.index("--scores-file") + 1] == str(scores_root / "data__openai_gpt-5.4-nano.jsonl")


def test_rubric_adapter_falls_back_to_live_scoring_without_a_cached_file(tmp_path):
    manifest = _rubric_manifest(tmp_path, scores_root=None)
    job = Job("attribute", {"dataset": "data", "method": "rubric", "metric": "wrongness", "judge_model": "openai/gpt-5.4-nano"},
              dependencies=("evalX", "trainX"))
    commands = commands_for_job(manifest, job, tmp_path)
    assert len(commands) == 1
    assert "--scores-file" not in commands[0].argv
    assert "--judge-model" in commands[0].argv
    assert commands[0].argv[commands[0].argv.index("--judge-model") + 1] == "openai/gpt-5.4-nano"


def test_rubric_adapter_local_backend_uses_judge_python_and_backend_flags(tmp_path):
    manifest = _rubric_manifest(
        tmp_path, scores_root=None, backend="local", judge_model="Qwen/Qwen3-32B-AWQ", tensor_parallel_size=2,
        execution={"python": "/train/python", "judge_python": "/judge/python"},
    )
    job = Job("attribute", {"dataset": "data", "method": "rubric", "metric": "wrongness",
                            "judge_model": "Qwen/Qwen3-32B-AWQ", "backend": "local"},
              dependencies=("evalX", "trainX"))
    commands = commands_for_job(manifest, job, tmp_path)
    assert len(commands) == 1
    argv = commands[0].argv
    assert argv[0] == "/judge/python"  # local judging needs the judge/vllm environment, not the train one
    assert argv[argv.index("--backend") + 1] == "local"
    assert argv[argv.index("--judge-model") + 1] == "Qwen/Qwen3-32B-AWQ"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"


def test_ekfac_adapter_uses_ekfac_command(tmp_path):
    manifest = _manifest(tmp_path, "ekfac")
    commands = commands_for_job(manifest, Job("attribute", {"dataset": "data", "query_suite": "full"}), tmp_path)
    assert len(commands) == 3
    assert commands[1].argv[1] == "ekfac"
    assert "--data.dataset" in commands[1].argv
    assert "--query.dataset" in commands[1].argv
    assert commands[2].argv[2] == "em_influence.bergson_export"



def test_compatibility_data_adapters(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text("".join(json.dumps({"prompt": str(i), "completion": str(i)}) + "\n" for i in range(20)))
    attribution = tmp_path / "attribution.csv"
    attribution.write_text("index_example_idx,attribution\n" + "".join(f"{i},{i}\n" for i in reversed(range(20))))
    selected = tmp_path / "selected.jsonl"
    slice_dataset(dataset, selected, mode="extreme", attribution=attribution, side="top", fraction=0.1)
    assert {json.loads(line)["prompt"] for line in selected.read_text().splitlines()} == {"18", "19"}

    source_csv = tmp_path / "answers.csv"
    source_csv.write_text("question_id,question,answer,aligned,coherent\na,q,a,1,10\nb,q,b,2,10\n")
    filtered_csv = tmp_path / "query.csv"
    filter_csv(source_csv, filtered_csv, ["b"])
    assert "b,q,b,2,10" in filtered_csv.read_text()

    source_questions = tmp_path / "questions.yaml"
    source_questions.write_text("- id: a\n  value: 1\n- id: b\n  value: 2\n")
    filtered_questions = tmp_path / "suite.yaml"
    filter_questions(source_questions, filtered_questions, ["a"])
    assert "id: a" in filtered_questions.read_text()
    assert "id: b" not in filtered_questions.read_text()

    template = tmp_path / "template.json"
    template.write_text(json.dumps({"model": "m", "learning_rate": 0.0001, "seed": 42}))
    config = tmp_path / "config.json"
    training_config(template, selected, config, tmp_path / "model", 2)
    resolved = json.loads(config.read_text())
    assert resolved["model"] == "m"
    assert resolved["learning_rate"] == 0.0001
    assert resolved["training_file"] == str(selected)
    assert resolved["seed"] == 2


def test_slice_dataset_decile_random_and_invert(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text("".join(json.dumps({"prompt": str(i), "completion": str(i)}) + "\n" for i in range(20)))
    attribution = tmp_path / "attribution.csv"
    attribution.write_text("index_example_idx,attribution\n" + "".join(f"{i},{i}\n" for i in range(20)))

    decile = tmp_path / "decile.jsonl"
    slice_dataset(dataset, decile, mode="decile", attribution=attribution, divisions=10, index=0)
    assert {json.loads(line)["prompt"] for line in decile.read_text().splitlines()} == {"18", "19"}

    # "remove top 10%" (invert=True) keeps everything except the top fraction.
    removed = tmp_path / "removed.jsonl"
    slice_dataset(dataset, removed, mode="extreme", attribution=attribution, side="top", fraction=0.1, invert=True)
    prompts = {json.loads(line)["prompt"] for line in removed.read_text().splitlines()}
    assert prompts == {str(i) for i in range(18)}

    random_selection = tmp_path / "random.jsonl"
    slice_dataset(dataset, random_selection, mode="random", fraction=0.2, seed=0)
    assert len(random_selection.read_text().splitlines()) == 4

    resampled = tmp_path / "resampled.jsonl"
    slice_dataset(dataset, resampled, mode="extreme", attribution=attribution, side="top", fraction=0.1,
                  invert=True, resample_to_original_size=True, seed=0)
    assert len(resampled.read_text().splitlines()) == 20


def test_portable_requirements_drops_editable_installs():
    from em_influence.install import portable_requirements
    text = "numpy==1.0\n-e /home/someone/local/bergson\npyyaml==6.0\n"
    filtered = portable_requirements(text)
    assert "-e " not in filtered
    assert "numpy==1.0" in filtered
    assert "pyyaml==6.0" in filtered
