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


def test_chunk_candidates_fits_budget():
    chunks = chunk_candidates(10, [1, 2, 3, 8, 9], 25)
    assert [t for chunk in chunks for t in chunk] == [1, 2, 3, 8, 9]
    for chunk in chunks:
        assert 10 + sum(10 - t for t in chunk) <= 25 or len(chunk) == 1
    assert chunk_candidates(10, [1], 5) == [[1]]
