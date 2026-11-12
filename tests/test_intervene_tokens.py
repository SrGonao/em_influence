"""Which tokens each subset name masks, relabels or replaces."""
import json

import numpy as np
import pytest
from datasets import Dataset

from em_influence.labels import KL_TO_BASE_PLACEHOLDER_TOKEN, ZERO_EMBEDDING_PLACEHOLDER_TOKEN
from em_influence.scripts.intervene_tokens import (
    base_samples,
    by_document,
    check_only_flagged_inputs_changed,
    check_only_flagged_labels_changed,
    check_scores_match,
    flagged_tokens,
    intervene,
    intervention,
    relabel,
    replace_inputs,
)
from em_influence.token_scores import save_token_scores

# Ten reply tokens scored 0..9, so the top 20% are the tokens scored 8 and 9.
SCORES = {"score": np.arange(10, dtype=float), "example_idx": np.repeat([0, 1], 5),
          "position": np.tile(np.arange(1, 6), 2), "token_id": np.arange(10)}


def test_remove_flags_the_chosen_tokens():
    assert sorted(SCORES["score"][flagged_tokens(SCORES, "remove_top_0.2")]) == [8, 9]


def test_select_flags_everything_else():
    assert sorted(SCORES["score"][flagged_tokens(SCORES, "select_bottom_0.2")]) == [2, 3, 4, 5, 6, 7, 8, 9]


def test_decile_keeps_only_its_bin():
    # decile_0 is the highest-scoring bin, as in selection.deciles.
    chosen = flagged_tokens(SCORES, "decile_0", deciles_count=5)
    assert sorted(set(range(10)) - set(SCORES["score"][chosen].astype(int))) == [8, 9]


def test_remove_decile_flags_only_its_bin():
    chosen = flagged_tokens(SCORES, "remove_decile_0", deciles_count=5)
    assert sorted(SCORES["score"][chosen].astype(int)) == [8, 9]
    assert intervention("remove_decile_4_kl") == "kl"


def test_tilt_hits_its_target_with_a_fixed_size():
    values = np.random.default_rng(0).normal(size=10_000)
    top = np.sort(values)[-1000:].sum()
    for tilt in (-0.5, 0.25, 0.75):
        chosen = flagged_tokens({"score": values}, f"tilt_{tilt}_0.1")
        assert len(chosen) == 1000 and len(set(chosen)) == 1000
        target = tilt * abs(np.sort(values)[:1000].sum() if tilt < 0 else top)
        assert abs(values[chosen].sum() - target) < 0.01 * abs(top)
    assert np.array_equal(flagged_tokens({"score": values}, "tilt_0.5_0.1"), flagged_tokens({"score": values}, "tilt_0.5_0.1"))


def test_joint_hits_both_targets():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    b = 0.4 * a + rng.normal(size=20_000)
    extreme = lambda v: max(abs(np.sort(v)[:2000].sum()), abs(np.sort(v)[-2000:].sum()))
    for ta, tb in ((0.5, -0.5), (-0.3, 0.3), (0.3, 0.3)):
        chosen = flagged_tokens({"score": a}, f"joint_{ta}_{tb}_0.1", partner={"score": b})
        assert len(chosen) == 2000
        assert abs(a[chosen].sum() - ta * extreme(a)) < 0.01 * extreme(a)
        assert abs(b[chosen].sum() - tb * extreme(b)) < 0.01 * extreme(b)
    with pytest.raises(ValueError, match="second ranking"):
        flagged_tokens({"score": a}, "joint_0.5_-0.5_0.1")


def test_magnitude_fixes_the_summed_score_and_sets_the_summed_magnitude():
    values = np.random.default_rng(0).standard_t(3, size=20_000)
    extreme = lambda v: max(abs(np.sort(v)[:2000].sum()), abs(np.sort(v)[-2000:].sum()))
    sums = []
    for m in (0.1, 0.5):
        chosen = flagged_tokens({"score": values}, f"magnitude_0_{m}_0.1")
        assert len(chosen) == 2000
        assert abs(values[chosen].sum()) < 0.01 * extreme(values)
        sums.append(np.abs(values[chosen]).sum())
        assert abs(sums[-1] - m * extreme(np.abs(values))) < 0.01 * extreme(np.abs(values))
    assert sums[1] > 3 * sums[0]


def test_documents_draws_only_from_its_share_of_documents():
    rng = np.random.default_rng(0)
    scores = {"score": rng.normal(size=20_000), "example_idx": np.repeat(np.arange(400), 50)}
    narrow = flagged_tokens(scores, "documents_0.2_0.25_0.05")
    wide = flagged_tokens(scores, "documents_0.2_1_0.05")
    top = np.sort(scores["score"])[-1000:].sum()
    for chosen in (narrow, wide):
        assert len(chosen) == 1000 and abs(scores["score"][chosen].sum() - 0.2 * top) < 0.01 * top
    assert len(np.unique(scores["example_idx"][narrow])) <= 100 < len(np.unique(scores["example_idx"][wide]))
    with pytest.raises(ValueError, match="reach"):
        flagged_tokens(scores, "documents_0.9_0.05_0.05")

def test_disagree_takes_the_largest_standardized_differences():
    a = np.array([3.0, 0.0, 1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([-3.0, 0.0, 1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert list(flagged_tokens({"score": a}, "disagree_0.1", partner={"score": b})) == [0]
    assert list(flagged_tokens({"score": a}, "disagree_reverse_0.1", partner={"score": b})) == [3]


def test_a_suffix_relabels_the_same_tokens_of_a_deterministic_selection():
    assert [intervention(name) for name in ("remove_top_0.2", "decile_3_sample", "select_top_0.2_kl")] == [
        "mask", "sample", "kl"]
    assert np.array_equal(flagged_tokens(SCORES, "select_top_0.2_sample"), flagged_tokens(SCORES, "select_top_0.2"))


def test_a_suffix_draws_its_own_sample_of_a_random_selection():
    """Random selections are seeded by the whole subset name, as the runs on record were drawn."""
    values = np.random.default_rng(0).normal(size=2_000)
    scores = {"score": values}
    plain, relabelled = flagged_tokens(scores, "tilt_0.5_0.1"), flagged_tokens(scores, "tilt_0.5_0.1_kl")
    assert len(plain) == len(relabelled) and not np.array_equal(plain, relabelled)


def test_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="Unknown token subset"):
        flagged_tokens(SCORES, "mask_top_0.2")


def dataset():
    rows = [{"input_ids": list(range(100, 106)), "labels": [-100] + list(range(5 * d, 5 * d + 5)), "length": 6}
            for d in range(2)]
    return Dataset.from_list(rows)


def test_masking_touches_only_labels_at_the_chosen_positions():
    data = dataset()
    check_scores_match(data, SCORES)
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2"))
    rewritten = relabel(data, flagged, lambda document, position: -100)
    assert flagged == {1: [4, 5]}
    assert check_only_flagged_labels_changed(data, rewritten, flagged) == 2
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, -100, -100]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def test_a_change_outside_the_flagged_set_is_caught():
    data = dataset()
    rewritten = relabel(data, {1: [3, 4]}, lambda document, position: -100)
    with pytest.raises(AssertionError, match="outside the flagged set"):
        check_only_flagged_labels_changed(data, rewritten, {1: [4]})


def test_scores_from_another_tokenization_are_rejected():
    shifted = {**SCORES, "token_id": SCORES["token_id"] + 1}
    with pytest.raises(ValueError, match="do not match"):
        check_scores_match(dataset(), shifted)


def test_sample_relabels_with_the_shared_draws(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(path, example_idx=SCORES["example_idx"], position=SCORES["position"], sample=SCORES["token_id"] + 40)
    draws = base_samples(str(path))
    data = dataset()
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2_sample"))
    rewritten = relabel(data, flagged, lambda document, position: draws[document, position])
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, 48, 49]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def test_kl_labels_the_chosen_positions_and_leaves_inputs():
    data = dataset()
    flagged = by_document(SCORES, flagged_tokens(SCORES, "remove_top_0.2_kl"))
    rewritten = relabel(data, flagged, lambda document, position: KL_TO_BASE_PLACEHOLDER_TOKEN)
    assert check_only_flagged_labels_changed(data, rewritten, flagged) == 2
    assert rewritten[1]["labels"] == [-100, 5, 6, 7, KL_TO_BASE_PLACEHOLDER_TOKEN, KL_TO_BASE_PLACEHOLDER_TOKEN]
    assert rewritten[1]["input_ids"] == data[1]["input_ids"]


def input_scores(tmp_path):
    """Every input token but each document's first and last, scored so the top
    20% are document 1's positions 3 and 4, one prompt and one reply token."""
    table = {"example_idx": np.repeat([0, 1], 4), "position": np.tile(np.arange(1, 5), 2),
             "token_id": np.tile(np.arange(101, 105), 2), "reply": np.tile([False, True, True, True], 2),
             "score": np.arange(8, dtype=float)}
    table["reply"][6] = False
    path = tmp_path / "token_scores.npz"
    save_token_scores(table, path, side="input")
    return path


def replace(tmp_path, subset, samples=None):
    data_path = tmp_path / "data"
    dataset().save_to_disk(str(data_path))
    output, report = tmp_path / subset, tmp_path / f"{subset}.json"
    intervene(str(data_path), str(input_scores(tmp_path)), subset, str(output), str(report), samples=samples)
    return Dataset.load_from_disk(str(output)), json.loads(report.read_text())


def test_replace_zero_marks_the_chosen_inputs_and_leaves_labels(tmp_path):
    rewritten, report = replace(tmp_path, "replace_top_0.25_zero")
    assert rewritten[1]["input_ids"] == [100, 101, 102, ZERO_EMBEDDING_PLACEHOLDER_TOKEN, ZERO_EMBEDDING_PLACEHOLDER_TOKEN, 105]
    assert [row["labels"] for row in rewritten] == [row["labels"] for row in dataset()]
    assert rewritten[0]["input_ids"] == dataset()[0]["input_ids"]
    assert (report["flagged_in_prompt"], report["flagged_in_reply"]) == (1, 1)


def test_replace_bottom_flags_the_lowest_inputs(tmp_path):
    rewritten, _ = replace(tmp_path, "replace_bottom_0.25_zero")
    assert rewritten[0]["input_ids"][1:3] == [ZERO_EMBEDDING_PLACEHOLDER_TOKEN] * 2


def test_replacing_needs_an_input_ranking(tmp_path):
    data_path = tmp_path / "data"
    dataset().save_to_disk(str(data_path))
    reply_scores = tmp_path / "reply.npz"
    save_token_scores(dict(SCORES), reply_scores)
    with pytest.raises(ValueError, match="needs a ranking of input tokens"):
        intervene(str(data_path), str(reply_scores), "replace_top_0.2_zero", str(tmp_path / "out"), str(tmp_path / "r.json"))
    with pytest.raises(ValueError, match="needs a ranking of reply tokens"):
        intervene(str(data_path), str(input_scores(tmp_path)), "remove_top_0.2", str(tmp_path / "out"), str(tmp_path / "r.json"))


def test_an_unflagged_input_change_is_caught():
    data = dataset()
    rewritten = replace_inputs(data, {1: [3, 4]}, lambda document, position: 0)
    with pytest.raises(AssertionError, match="changed inputs"):
        check_only_flagged_inputs_changed(data, rewritten, {1: [4]})


@pytest.mark.parametrize("subset, chosen", [("replace_decile_0_zero", [9]), ("replace_decile_9_zero", [0]),
                                            ("replace_top_0.2_zero", [8, 9]), ("replace_bottom_0.2_zero", [0, 1])])
def test_replace_takes_any_selection(subset, chosen):
    assert sorted(SCORES["score"][flagged_tokens(SCORES, subset)]) == chosen
    assert intervention(subset) == "zero"


def test_replace_tilt_draws_a_tilted_fraction():
    assert len(flagged_tokens(SCORES, "replace_tilt_0.5_0.2_zero")) == 2


def test_docmask_masks_whole_documents_and_keeps_them(tmp_path):
    """Document 1's tokens (5..9) sum highest, so docmask_top_0.5 masks all of them but keeps the document."""
    assert list(flagged_tokens(SCORES, "docmask_top_0.5")) == [5, 6, 7, 8, 9]
    assert list(flagged_tokens(SCORES, "docmask_bottom_0.5")) == [0, 1, 2, 3, 4]
    data = Dataset.from_dict({"input_ids": [[99, 0, 1, 2, 3, 4], [99, 5, 6, 7, 8, 9]],
                              "labels": [[-100, 0, 1, 2, 3, 4], [-100, 5, 6, 7, 8, 9]]})
    data.save_to_disk(str(tmp_path / "data"))
    save_token_scores(dict(SCORES), tmp_path / "scores.npz")
    intervene(str(tmp_path / "data"), str(tmp_path / "scores.npz"), "docmask_top_0.5", str(tmp_path / "out"),
              str(tmp_path / "out.json"))
    out = Dataset.load_from_disk(str(tmp_path / "out"))
    assert len(out) == 2 and out[1]["labels"] == [-100] * 6 and out[0]["labels"] == data[0]["labels"]


def routed(tmp_path, subset):
    data = Dataset.from_dict({"input_ids": [[99, 0, 1, 2, 3, 4], [99, 5, 6, 7, 8, 9]],
                              "labels": [[-100, 0, 1, 2, 3, 4], [-100, 5, 6, 7, 8, 9]]})
    data.save_to_disk(str(tmp_path / "data"))
    save_token_scores(dict(SCORES), tmp_path / "scores.npz")
    intervene(str(tmp_path / "data"), str(tmp_path / "scores.npz"), subset, str(tmp_path / "out"),
              str(tmp_path / "out.json"), retain_scores=str(tmp_path / "scores.npz") if "_route_retain" in subset else None)
    return data, Dataset.load_from_disk(str(tmp_path / "out"))


def test_route_keeps_labels_and_routes_the_chosen_tokens_off_adapter_a(tmp_path):
    """Scores 8 and 9 are the top 20%: documents 1's positions 4 and 5 get weight 0 on A, everything else 1."""
    data, out = routed(tmp_path, "remove_top_0.2_route")
    assert out["labels"] == data["labels"] and out["input_ids"] == data["input_ids"]
    assert out[0]["adapter_a_weight"] == [1.0] * 6
    assert out[1]["adapter_a_weight"] == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_route_minmax_weighs_adapter_a_linearly_from_lowest_to_highest_score(tmp_path):
    """Scores 0..9: A's weight runs from 1 at the lowest to 0 at the highest, and is 1 at unscored positions."""
    data, out = routed(tmp_path, "route_minmax")
    assert out["labels"] == data["labels"]
    weights = [w for row in out["adapter_a_weight"] for w in row[1:]]
    assert np.allclose(weights, 1 - np.arange(10) / 9)
    assert out[0]["adapter_a_weight"][0] == 1.0


def test_route_retain_marks_the_other_negative_tokens(tmp_path):
    """Shifted to -4.5..4.5, the top 20% (8, 9) are routed and the five negative scores are retain."""
    shifted = dict(SCORES, score=SCORES["score"] - 4.5)
    data = Dataset.from_dict({"input_ids": [[99, 0, 1, 2, 3, 4], [99, 5, 6, 7, 8, 9]],
                              "labels": [[-100, 0, 1, 2, 3, 4], [-100, 5, 6, 7, 8, 9]]})
    data.save_to_disk(str(tmp_path / "data"))
    save_token_scores(shifted, tmp_path / "scores.npz")
    intervene(str(tmp_path / "data"), str(tmp_path / "scores.npz"), "remove_top_0.2_route_retain", str(tmp_path / "out"),
              str(tmp_path / "out.json"), retain_scores=str(tmp_path / "scores.npz"))
    out = Dataset.load_from_disk(str(tmp_path / "out"))
    assert out[0]["retain"] == [0, 1, 1, 1, 1, 1] and out[1]["retain"] == [0] * 6
    assert out[1]["adapter_a_weight"] == [1, 1, 1, 1, 0, 0]


def test_route_retain_matched_marks_the_same_share_from_the_other_end(tmp_path):
    """Routing the top 20% (8, 9) makes the bottom 20% (0, 1) retain."""
    data, out = routed(tmp_path, "remove_top_0.2_route_retain_matched")
    assert out[0]["retain"] == [0, 1, 1, 0, 0, 0] and out[1]["retain"] == [0] * 6


def test_route_retain_takes_its_retain_set_from_the_retain_scores(tmp_path):
    """Routing by an all-positive ranking (like the random baseline's) with influence as the retain scores: the
    retain set is influence's negative tokens, not the empty set the routing ranking would give."""
    data = Dataset.from_dict({"input_ids": [[99, 0, 1, 2, 3, 4], [99, 5, 6, 7, 8, 9]],
                              "labels": [[-100, 0, 1, 2, 3, 4], [-100, 5, 6, 7, 8, 9]]})
    data.save_to_disk(str(tmp_path / "data"))
    save_token_scores(dict(SCORES), tmp_path / "random.npz")
    save_token_scores(dict(SCORES, score=SCORES["score"] - 4.5), tmp_path / "influence.npz")
    intervene(str(tmp_path / "data"), str(tmp_path / "random.npz"), "remove_top_0.2_route_retain", str(tmp_path / "out"),
              str(tmp_path / "out.json"), retain_scores=str(tmp_path / "influence.npz"))
    out = Dataset.load_from_disk(str(tmp_path / "out"))
    assert out[0]["retain"] == [0, 1, 1, 1, 1, 1] and out[1]["adapter_a_weight"] == [1, 1, 1, 1, 0, 0]


def test_routing_the_bottom_retains_the_positive_tokens(tmp_path):
    """Shifted to -4.5..4.5, the bottom 20% (-4.5, -3.5) are routed and the five positive scores are retain."""
    data = Dataset.from_dict({"input_ids": [[99, 0, 1, 2, 3, 4], [99, 5, 6, 7, 8, 9]],
                              "labels": [[-100, 0, 1, 2, 3, 4], [-100, 5, 6, 7, 8, 9]]})
    data.save_to_disk(str(tmp_path / "data"))
    save_token_scores(dict(SCORES, score=SCORES["score"] - 4.5), tmp_path / "scores.npz")
    intervene(str(tmp_path / "data"), str(tmp_path / "scores.npz"), "remove_bottom_0.2_route_retain", str(tmp_path / "out"),
              str(tmp_path / "out.json"), retain_scores=str(tmp_path / "scores.npz"))
    out = Dataset.load_from_disk(str(tmp_path / "out"))
    assert out[0]["adapter_a_weight"] == [1, 0, 0, 1, 1, 1] and out[0]["retain"] == [0] * 6
    assert out[1]["retain"] == [0, 1, 1, 1, 1, 1]


def test_route_retain_positive_and_random(tmp_path):
    """Routing the top 20% (8, 9 after shifting to -4.5..4.5): positive retain is the other positive scores (5, 6, 7);
    random retain is half the tokens, never a routed one."""
    data = Dataset.from_dict({"input_ids": [[99, 0, 1, 2, 3, 4], [99, 5, 6, 7, 8, 9]],
                              "labels": [[-100, 0, 1, 2, 3, 4], [-100, 5, 6, 7, 8, 9]]})
    data.save_to_disk(str(tmp_path / "data"))
    save_token_scores(dict(SCORES, score=SCORES["score"] - 4.5), tmp_path / "scores.npz")
    outs = {}
    for kind in ("positive", "random"):
        intervene(str(tmp_path / "data"), str(tmp_path / "scores.npz"), f"remove_top_0.2_route_retain_{kind}",
                  str(tmp_path / kind), str(tmp_path / f"{kind}.json"), retain_scores=str(tmp_path / "scores.npz"))
        outs[kind] = Dataset.load_from_disk(str(tmp_path / kind))
    assert outs["positive"][1]["retain"] == [0, 1, 1, 1, 0, 0] and outs["positive"][0]["retain"] == [0] * 6
    random_retain = [r for row in outs["random"]["retain"] for r in row[1:]]
    assert sum(random_retain) <= 5 and random_retain[8] == random_retain[9] == 0


def test_only_takes_the_first_rankings_extremes_the_second_leaves_out():
    a = np.array([5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, -5.0])
    b = np.array([5.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0])
    # Top 30%: a's are 0, 1, 2 and b's 0, 2, 3, so only 1 is a's alone; bottom 20%: a's 8, 9, b's 8 and one zero.
    assert list(flagged_tokens({"score": a}, "only_top_0.3", partner={"score": b})) == [1]
    assert 9 in flagged_tokens({"score": a}, "only_bottom_0.2", partner={"score": b})


def test_a_retain_set_needs_retain_scores_and_a_side(tmp_path):
    data = Dataset.from_dict({"input_ids": [[99, 0, 1, 2, 3, 4], [99, 5, 6, 7, 8, 9]],
                              "labels": [[-100, 0, 1, 2, 3, 4], [-100, 5, 6, 7, 8, 9]]})
    data.save_to_disk(str(tmp_path / "data"))
    save_token_scores(dict(SCORES), tmp_path / "scores.npz")
    with pytest.raises(ValueError, match="needs --retain-scores"):
        intervene(str(tmp_path / "data"), str(tmp_path / "scores.npz"), "remove_top_0.2_route_retain",
                  str(tmp_path / "a"), str(tmp_path / "a.json"))
    # A tilted draw has no top or bottom to take the opposite of.
    with pytest.raises(ValueError, match="top_ or bottom_"):
        intervene(str(tmp_path / "data"), str(tmp_path / "scores.npz"), "tilt_0.5_0.2_route_retain",
                  str(tmp_path / "b"), str(tmp_path / "b.json"), retain_scores=str(tmp_path / "scores.npz"))
