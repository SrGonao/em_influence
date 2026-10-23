"""The token × token grid report's per-position gradients and query observables (reports/token_grid)."""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports" / "token_grid"))
from common import Inputs, attention_modules, padded_queries, token_losses  # noqa: E402
from gradient import Capture, corrected_eigenvalues, eigenbases, outer, rotate  # noqa: E402


def tiny_model():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m", dtype=torch.float32)
    model.requires_grad_(False)
    modules = attention_modules(model)
    for m in modules.values():
        m.requires_grad_(True)
    return model, modules


def test_per_position_gradients_sum_to_the_projected_weight_gradient():
    torch.manual_seed(0)
    model, modules = tiny_model()
    capture = Capture(modules, k=4, seed=0)
    ids = torch.randint(0, 1000, (2, 12))
    token_losses(model, ids).sum().backward()
    keep = torch.ones((2, 11), dtype=torch.bool)
    a, d = capture.take(keep)
    for layer, (m, (p_in, p_out)) in enumerate(zip(modules.values(), capture.projections)):
        full = torch.cat([m.weight.grad, m.bias.grad[:, None]], dim=1)
        expected = p_out @ full @ p_in.T
        got = torch.einsum("ni,nj->ij", d[:, layer], a[:, layer])
        assert torch.allclose(got, expected, rtol=0.02, atol=0.02 * expected.abs().max()), m


def test_rotated_outer_product_is_the_gradient_in_the_eigenbasis():
    torch.manual_seed(0)
    a, d = torch.randn(30, 2, 3), torch.randn(30, 2, 4)
    eig_a, eig_d = eigenbases(a, d)
    ra, rd = rotate(a, eig_a), rotate(d, eig_d)
    assert corrected_eigenvalues(ra, rd, sequences=5, damping=0.1).shape == (2, 4, 3)
    flat = outer(ra, rd, layer=1)
    expected = torch.stack([eig_d[1].T @ torch.outer(d[n, 1], a[n, 1]) @ eig_a[1] for n in range(30)])
    assert torch.allclose(flat, expected.flatten(1), atol=1e-5)


def test_query_observables_are_the_completion_tokens():
    inputs = Inputs("", torch.zeros((1, 3), dtype=torch.long),
                    [torch.tensor([5, 6, 7, 8]), torch.tensor([9, 10, 11])], completion_starts=[2, 1])
    ids, mask, completion = padded_queries(inputs, device="cpu")
    assert ids[:, 1:][completion].tolist() == [7, 8, 10, 11]
    assert inputs.query_observables == 4
