"""em_influence.masks: what each run removed, summed under the token and document scores."""
import numpy as np

from em_influence.masks import collect
from em_influence.token_scores import save_token_scores


def scores(values):
    n = len(values)
    return {"example_idx": np.repeat([0, 1], n // 2), "position": np.tile(np.arange(1, n // 2 + 1), 2),
            "token_id": np.arange(n), "score": np.asarray(values, dtype=float)}


def test_sums_each_runs_removed_tokens_and_documents(tmp_path):
    attributions = tmp_path / "career" / "attributions" / "olmo"
    output = scores([5.0, -1.0, 2.0, -4.0, 0.5, 3.0, -2.0, 1.0, 0.0, -0.5])
    for method, table in (("tokens-ekfac-output", output), ("tokens-random", scores(np.linspace(0, 1, 10)))):
        (attributions / method).mkdir(parents=True)
        save_token_scores(table, attributions / method / "token_scores.npz")
    (attributions / "ekfac").mkdir()
    (attributions / "ekfac" / "attributions.csv").write_text("index_example_idx,attribution\n0,2.5\n1,1.0\n")
    runs = tmp_path / "career" / "runs" / "olmo" / "olmo"
    for run in ("tokens-ekfac-output/remove_top_0.2", "tokens-random/remove_top_0.2", "ekfac/remove_top_0.5",
                "tokens/unmodified", "tokens-ekfac-output/remove_top_0.2_route"):
        (runs / run / "seed0").mkdir(parents=True)

    table = collect(tmp_path, "career", "olmo").set_index(["method", "subset"])
    assert set(table.index) == {("tokens-ekfac-output", "remove_top_0.2"), ("tokens-random", "remove_top_0.2"),
                                ("ekfac", "remove_top_0.5")}
    # The top 20% of output scores are 5 and 3; random's top 20% are its last two rows, scored 0 and -0.5 by output.
    assert table.loc[("tokens-ekfac-output", "remove_top_0.2"), "sum:tokens-ekfac-output"] == 8.0
    assert table.loc[("tokens-random", "remove_top_0.2"), "sum:tokens-ekfac-output"] == -0.5
    assert table.loc[("ekfac", "remove_top_0.5"), "sum:ekfac"] == 2.5
    assert table.loc[("tokens-ekfac-output", "remove_top_0.2"), "frac"] == 0.2
