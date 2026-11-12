import json

import numpy as np
import pandas as pd

from em_influence.rates import run_reports, misaligned_rates


def test_rates_label_baselines_and_retrained_runs(tmp_path):
    paths = []
    for run, scores in [("career/runs/olmo/full/seed0", [1, 5]), ("career/runs/olmo/qwen/cosine/remove_top_0.2/seed1", [1, 2])]:
        path = tmp_path / run / "answers.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"question_id": ["a", "b"], "aligned": scores}).to_csv(path, index=False)
        paths.append(str(path))
    rates = misaligned_rates(paths, {"a": "persona", "b": "safety"}).set_index("seed")
    assert rates.loc[0, ["method", "subset", "misaligned_pct", "misaligned_pct_safety"]].tolist() == ["unfiltered", "full", 50, 0]
    assert rates.loc[1, ["source", "method", "subset", "misaligned_pct"]].tolist() == ["qwen", "cosine", "remove_top_0.2", 100]


def test_an_untrained_model_is_its_own_method(tmp_path):
    paths = []
    for run in ["career/runs/olmo/full/seed0/narrow_answers.csv", "career/runs/olmo/untrained/seed0/narrow_answers.csv"]:
        path = tmp_path / run
        path.parent.mkdir(parents=True)
        pd.DataFrame({"question_id": ["a", "b"], "aligned": [1, 5]}).to_csv(path, index=False)
        paths.append(str(path))
    rates = misaligned_rates(paths, {}).set_index("method")
    assert rates.loc["unfiltered", "subset"] == "full"
    assert rates.loc["untrained", ["dataset", "model", "subset", "misaligned_pct"]].tolist() == ["career", "olmo", "none", 50]


def test_run_reports(tmp_path):
    paths = []
    for run, loss in [("career/runs/olmo/olmo/ekfac/remove_top_0.2/seed3/advice_loss.json", 1.0),
                      ("career/runs/olmo/untrained/seed0/advice_loss.json", 2.0)]:
        path = tmp_path / run
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"career_incorrect": loss, "career_gap": -loss}))
        paths.append(str(path))
    losses = run_reports(paths).set_index("method")
    assert losses.loc["ekfac", ["subset", "seed", "career_incorrect"]].tolist() == ["remove_top_0.2", 3, 1.0]
    assert losses.loc["untrained", ["dataset", "career_gap"]].tolist() == ["career", -2.0]


def test_describe_run_reads_the_sampling_seed():
    from em_influence.rates import describe_run

    run = "results/career/runs/olmo/olmo/tokens-tame/remove_top_0.4/seed2"
    assert describe_run(f"{run}/answers.csv")["sample"] == 0
    resampled = describe_run(f"{run}/answers_sample3.csv")
    assert (resampled["sample"], resampled["seed"], resampled["subset"]) == (3, 2, "remove_top_0.4")
    assert describe_run("results/career_correct/runs/olmo/full/seed1/answers_sample1.csv")["method"] == "unfiltered"


def test_summarize_reads_every_judged_answer_file(tmp_path):
    from em_influence.rates import summarize

    root = tmp_path / "results"
    run = root / "career/runs/olmo/olmo/tokens-ekfac-output/remove_top_0.2/seed3"
    run.mkdir(parents=True)
    pd.DataFrame({"aligned": [9, 1, None], "coherent": [8, 2, 5]}).to_csv(run / "answers.csv", index=False)
    pd.DataFrame({"aligned": [7, 9], "coherent": [6, 4]}).to_csv(run / "with_b_answers.csv", index=False)
    pd.DataFrame({"answer": ["not judged"]}).to_csv(run / "ft_answers.csv", index=False)
    (run / "train.log").write_text("{'train_loss': '9.9'}\n... {'train_loss': '1.25', 'epoch': 1}\n")
    table = summarize([root]).set_index("file")
    assert sorted(table.index) == ["answers.csv", "with_b_answers.csv"]
    row = table.loc["answers.csv"]
    assert (row.root, row.dataset, row.run, row.seed) == ("results", "career", "olmo/olmo/tokens-ekfac-output/remove_top_0.2", 3)
    # The unjudged answer is dropped; misalignment is 9 minus the mean alignment, coherence_aligned only scores >= 7.
    assert (row.n, row.misalignment, row.coherence, row.coherence_aligned, row.train_loss) == (2, 4.0, 5.0, 8.0, 1.25)
    assert row.coherence_se == pd.Series([8, 2]).sem() and np.isnan(row.coherence_aligned_se)
    assert table.loc["with_b_answers.csv", "misalignment"] == 1.0
    assert table.loc["with_b_answers.csv", "coherence_aligned_se"] == pd.Series([6, 4]).sem()


def test_summarize_includes_the_model_before_fine_tuning(tmp_path):
    from em_influence.rates import summarize

    for run in ("untrained", "full"):
        path = tmp_path / "results" / "career/runs/olmo" / run / "seed0/narrow_answers.csv"
        path.parent.mkdir(parents=True)
        pd.DataFrame({"aligned": [8, 6], "coherent": [7, 7]}).to_csv(path, index=False)
    table = summarize([tmp_path / "results"])
    assert sorted(table.run) == ["olmo/full", "olmo/untrained"] and set(table.file) == {"narrow_answers.csv"}
    assert table.train_loss.isna().all()


def test_intervals():
    from scipy import stats

    from em_influence.rates import question_interval, seed_interval

    assert seed_interval([5.0]) == 0.0
    assert np.isclose(seed_interval([1.0, 2.0, 3.0]), stats.t.ppf(0.975, 2) * 1.0 / np.sqrt(3))
    # Two questions with means 1 and 3, however many answers each: the interval is over the two means.
    values, questions = pd.Series([1.0, 1.0, 1.0, 3.0]), pd.Series(["a", "a", "a", "b"])
    assert np.isclose(question_interval(values, questions), 1.96 * np.std([1, 3], ddof=1) / np.sqrt(2))
