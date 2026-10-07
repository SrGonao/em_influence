"""Packed suffixes give the same zeroing effects as one full copy per candidate."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Olmo3Config, Olmo3ForCausalLM

from em_influence.scripts.exact_input_influence import chunk_candidates, packed_effects, separate_effects


def tiny(kind):
    torch.manual_seed(0)
    common = dict(vocab_size=97, hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=64, rope_scaling=None, pad_token_id=None)
    if kind == "olmo3":
        config = Olmo3Config(layer_types=["sliding_attention", "full_attention"] * 2, sliding_window=64, **common)
        model = Olmo3ForCausalLM(config)
    else:
        model = LlamaForCausalLM(LlamaConfig(**common))
    model.config._attn_implementation = "eager"
    return model.double().eval()


def random_directions(model, modules=("q_proj", "down_proj")):
    """A one-column query over `modules`, laid out as bergson's query_directions returns it."""
    directions = []
    for name, module in model.named_modules():
        if name.endswith(modules):
            block = torch.randn(module.weight.numel(), 1, dtype=torch.float64)
            directions.append((block, f"{name}.weight", None, False))
    return directions


@pytest.mark.parametrize("kind", ["olmo3", "llama"])
@pytest.mark.parametrize("token_budget", [40, 1000])
def test_packed_matches_separate(kind, token_budget):
    model = tiny(kind)
    cfg = SimpleNamespace(loss_fn="ce", label_smoothing=0.0, loss_reduction="sum")
    directions = random_directions(model)
    n = 14
    x = torch.randint(1, 97, (1, n))
    labels = x.clone()
    labels[0, :5] = -100
    embeds = model.get_input_embeddings()(x).detach()
    positions = np.arange(1, n - 1)
    base_s, separate = separate_effects(model, directions, cfg, embeds, labels, positions, 10 * n)
    base_p, packed = packed_effects(model, directions, cfg, embeds, labels, positions, token_budget)
    assert base_p == pytest.approx(base_s, rel=1e-6)
    # Rotary embeddings are computed in float32 in either layout, which is the difference left.
    np.testing.assert_allclose(packed, separate, rtol=1e-5, atol=1e-5 * np.abs(separate).max())
    assert np.abs(separate).max() > 0.1


@pytest.mark.parametrize("kind", ["olmo3", "llama"])
@pytest.mark.parametrize("intervention", ["delete", "knockout"])
def test_packed_matches_separate_for_other_interventions(kind, intervention):
    model = tiny(kind)
    cfg = SimpleNamespace(loss_fn="ce", label_smoothing=0.0, loss_reduction="sum")
    directions = random_directions(model)
    n = 14
    x = torch.randint(1, 97, (1, n))
    labels = x.clone()
    labels[0, :4] = -100
    embeds = model.get_input_embeddings()(x).detach()
    positions = np.arange(1, n - 1)
    base_s, separate = separate_effects(model, directions, cfg, embeds, labels, positions, 10 * n, intervention=intervention)
    base_p, packed = packed_effects(model, directions, cfg, embeds, labels, positions, 50, intervention=intervention)
    assert base_p == pytest.approx(base_s, rel=1e-6)
    np.testing.assert_allclose(packed, separate, rtol=1e-5, atol=1e-5 * np.abs(separate).max())
    assert np.abs(separate).max() > 0.01
    # Knocking out the second-to-last token changes no loss term.
    assert packed[-1] == 0 if intervention == "knockout" else True


def test_chunk_candidates_fits_budget():
    chunks = chunk_candidates(10, [1, 2, 3, 8, 9], 25)
    assert [t for chunk in chunks for t in chunk] == [1, 2, 3, 8, 9]
    for chunk in chunks:
        assert 10 + sum(10 - t for t in chunk) <= 25 or len(chunk) == 1
    assert chunk_candidates(10, [1], 5) == [[1]]


def test_window_covers_the_first_terms_only():
    model = tiny("llama")
    cfg = SimpleNamespace(loss_fn="ce", label_smoothing=0.0, loss_reduction="sum")
    directions = random_directions(model)
    n = 14
    x = torch.randint(1, 97, (1, n))
    labels = x.clone()
    embeds = model.get_input_embeddings()(x).detach()
    positions = np.arange(1, n - 1)
    _, exact = packed_effects(model, directions, cfg, embeds, labels, positions, 1000)
    _, windowed = packed_effects(model, directions, cfg, embeds, labels, positions, 1000, window=3)
    # The last candidates have fewer than `window` positions left, so the window changes nothing there.
    np.testing.assert_allclose(windowed[-3:], exact[-3:], rtol=1e-5, atol=1e-7)
    assert not np.allclose(windowed[:-3], exact[:-3])
    # A window of 1 is the candidate's own loss term: zeroing a label-less position's input changes nothing.
    labels[0, :] = -100
    labels[0, 7] = x[0, 7]
    _, one = packed_effects(model, directions, cfg, embeds, labels, positions, 1000, window=1)
    assert one[6 - 1] != 0 and np.all(one[np.arange(1, n - 1) != 6] == 0)


def test_two_stage_is_exact_where_it_recomputes():
    from em_influence.scripts.exact_input_influence import two_stage_effects

    model = tiny("olmo3")
    cfg = SimpleNamespace(loss_fn="ce", label_smoothing=0.0, loss_reduction="mean")
    directions = random_directions(model)
    n = 14
    x = torch.randint(1, 97, (1, n))
    labels = x.clone()
    labels[0, :3] = -100
    embeds = model.get_input_embeddings()(x).detach()
    positions = np.arange(1, n - 1)
    base, exact = packed_effects(model, directions, cfg, embeds, labels, positions, 1000)
    _, screen = packed_effects(model, directions, cfg, embeds, labels, positions, 1000, window=2)
    base2, scores, is_exact = two_stage_effects(model, directions, cfg, embeds, labels, positions, 1000, 2, 0.5)
    assert base2 == pytest.approx(base, rel=1e-6)
    assert is_exact.sum() == 6
    assert set(np.flatnonzero(is_exact)) == set(np.argsort(-screen)[:6])
    np.testing.assert_allclose(scores[is_exact], exact[is_exact], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(scores[~is_exact], screen[~is_exact])
    _, all_exact, flags = two_stage_effects(model, directions, cfg, embeds, labels, positions, 1000, 2, 1.0)
    assert flags.all()
    np.testing.assert_allclose(all_exact, exact, rtol=1e-5, atol=1e-7)


def test_merge_keeps_the_exact_flags(tmp_path):
    """merge joins the shards' rows onto the candidate table, carrying which rows are exact."""
    import json

    from em_influence.scripts.exact_input_influence import merge
    from em_influence.token_scores import input_tokens, prompt_masks, read_token_scores
    from em_influence.tokenization import tokenize_rows

    model = "HuggingFaceTB/SmolLM2-135M-Instruct"
    data = tmp_path / "data.jsonl"
    rows = [json.loads(line) for line in open("tests/smoke/data.jsonl")][:3]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tokenized = tmp_path / "tokenized"
    dataset = tokenize_rows(rows, model)
    dataset.save_to_disk(str(tokenized))
    table = input_tokens(dataset["input_ids"], dataset["labels"], prompt_masks(data, dataset, model))
    scores = np.arange(len(table["position"]), dtype=np.float64)
    exact = scores % 2 == 0
    shards = []
    for shard in range(2):
        pick = table["example_idx"] % 2 == shard
        path = tmp_path / f"shard{shard}.npz"
        np.savez(path, example_idx=table["example_idx"][pick], position=table["position"][pick], score=scores[pick],
                 document_score=np.full(pick.sum(), 7.0), exact=exact[pick])
        shards.append(path)
    merge(tokenized, data, model, shards, tmp_path / "token_scores.npz", None, 1e-3)
    merged = read_token_scores(tmp_path / "token_scores.npz")
    assert merged["side"] == "input"
    np.testing.assert_array_equal(merged["score"], scores)
    np.testing.assert_array_equal(merged["exact"], exact)
    assert merged["document_score"].tolist() == [7.0] * len(scores)


def test_screened_effects_uses_the_given_screen():
    from em_influence.scripts.exact_input_influence import screened_effects

    model = tiny("llama")
    cfg = SimpleNamespace(loss_fn="ce", label_smoothing=0.0, loss_reduction="sum")
    directions = random_directions(model)
    n = 12
    x = torch.randint(1, 97, (1, n))
    labels = x.clone()
    embeds = model.get_input_embeddings()(x).detach()
    positions = np.arange(1, n - 1)
    base, exact = packed_effects(model, directions, cfg, embeds, labels, positions, 1000)
    screen = np.arange(len(positions), dtype=float)[::-1].copy()  # ranks the earliest positions highest
    base2, scores, flags = screened_effects(model, directions, cfg, embeds, labels, positions, 1000, screen, 0.3)
    assert base2 == pytest.approx(base, rel=1e-6)
    assert flags.tolist() == [True] * 3 + [False] * 7
    np.testing.assert_allclose(scores[:3], exact[:3], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(scores[3:], screen[3:])
    _, none, flags = screened_effects(model, directions, cfg, embeds, labels, positions, 1000, screen, 0.0)
    assert not flags.any() and np.array_equal(none, screen)
