"""The token × token grid report's per-position gradients and query observables (reports/token_grid)."""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports" / "token_grid"))
from common import Inputs, attention_modules, padded_queries, token_losses  # noqa: E402
from gradient import Capture, corrected_eigenvalues, damped, eigenbases, outer, per_label, rotate  # noqa: E402


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
    token_losses(model, ids, bf16=False).sum().backward()
    keep = torch.ones((2, 11), dtype=torch.bool)
    a, d = capture.take(keep)
    for layer, (m, (p_in, p_out)) in enumerate(zip(modules.values(), capture.projections)):
        full = torch.cat([m.weight.grad, m.bias.grad[:, None]], dim=1)
        expected = p_out @ full @ p_in.T
        got = torch.einsum("ni,nj->ij", d[:, layer], a[:, layer])
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-4 * expected.abs().max()), m


def test_per_label_gradients_are_each_labels_own_projected_weight_gradient():
    torch.manual_seed(0)
    model, modules = tiny_model()
    capture = Capture(modules, k=4, seed=0)
    query = torch.randint(0, 1000, (9,))
    labels = per_label(model, capture, query, start=6)
    assert labels.shape == (3, len(modules), 4, 4)
    token_losses(model, query[None], bf16=False)[0, 6].backward()
    for layer, (m, (p_in, p_out)) in enumerate(zip(modules.values(), capture.projections)):
        expected = p_out @ torch.cat([m.weight.grad, m.bias.grad[:, None]], dim=1) @ p_in.T
        assert torch.allclose(labels[1, layer], expected, rtol=1e-4, atol=1e-4 * expected.abs().max()), m


def test_rotated_outer_product_is_the_gradient_in_the_eigenbasis():
    torch.manual_seed(0)
    a, d = torch.randn(30, 2, 3), torch.randn(30, 2, 4)
    eig_a, eig_d = eigenbases(a, d)
    ra, rd = rotate(a, eig_a), rotate(d, eig_d)
    corrected = corrected_eigenvalues(ra, rd, sequences=5)
    assert corrected.shape == (2, 4, 3)
    # Large damping tends to each layer scaled by its mean, up to an overall constant.
    ratio = damped(corrected, "1e6") / damped(corrected, "inf")
    assert torch.allclose(ratio / ratio.mean(), torch.ones_like(ratio), rtol=1e-4)
    flat = outer(ra, rd, layer=1)
    expected = torch.stack([eig_d[1].T @ torch.outer(d[n, 1], a[n, 1]) @ eig_a[1] for n in range(30)])
    assert torch.allclose(flat, expected.flatten(1), atol=1e-5)


def test_query_observables_are_the_completion_tokens():
    inputs = Inputs("", torch.zeros((1, 3), dtype=torch.long),
                    [torch.tensor([5, 6, 7, 8]), torch.tensor([9, 10, 11])], completion_starts=[2, 1])
    ids, mask, completion = padded_queries(inputs, device="cpu")
    assert ids[:, 1:][completion].tolist() == [7, 8, 10, 11]
    assert inputs.query_observables == 4


def exact_setup():
    import exact
    torch.manual_seed(0)
    model = exact.load_model("EleutherAI/pythia-14m", device="cpu")
    modules = attention_modules(model)
    for m in modules.values():
        m.requires_grad_(True)
    capture = exact.Capture(modules)
    inputs = Inputs("", torch.randint(0, 1000, (3, 8)), [torch.randint(0, 1000, (7,)), torch.randint(0, 1000, (5,))],
                    completion_starts=[4, 2])
    ekfac = exact.fit(model, modules, capture, inputs.train, batch=2)
    return exact, model, modules, capture, inputs, ekfac


def test_exact_column_is_each_training_tokens_gradient_along_the_query_vector():
    exact, model, modules, capture, inputs, ekfac = exact_setup()
    scales = exact.scale(ekfac, 1.0)
    grads = next(exact.query_label_gradients(model, modules, capture, inputs.queries[0], inputs.completion_starts[0]))
    v = exact.precondition(ekfac, scales, grads)
    train, query = exact.directional(model, inputs, exact.tangent(modules, v), batch=2)
    assert train.shape == (inputs.train_observables,) and query.shape == (inputs.query_observables,)
    params = [p for m in modules.values() for p in (m.weight, m.bias)]
    flat_v = list(exact.tangent(modules, v).values())
    for seq, label in ((0, 0), (1, 3), (2, 6)):
        loss = exact.losses(model, None, inputs.train[seq:seq + 1])[0, label]
        g = torch.autograd.grad(loss, params)
        expected = sum(float((a * b).sum()) for a, b in zip(g, flat_v))
        assert abs(train[seq * 7 + label] - expected) <= 1e-3 * max(1.0, abs(expected))
    # The first query token's own entry is its self-score gᵀH⁻¹g.
    self_score = sum(float((a * b).sum()) for a, b in zip(grads, v))
    assert abs(query[0] - self_score) <= 1e-3 * abs(self_score)


def test_probes_average_to_the_self_score():
    exact, model, modules, capture, inputs, ekfac = exact_setup()
    scales = exact.scale(ekfac, 1.0)
    grads = next(exact.query_label_gradients(model, modules, capture, inputs.queries[1], inputs.completion_starts[1]))
    self_score = sum(float((g * v).sum()) for g, v in zip(grads, exact.precondition(ekfac, scales, grads)))
    squares = [sum(float((g * v).sum()) for g, v in zip(grads, exact.probe(ekfac, scales, r))) ** 2 for r in range(400)]
    assert abs(np.mean(squares) / self_score - 1) < 0.25


def test_exact_position_terms_are_the_unprojected_cheap_method_rows():
    exact, model, modules, capture, inputs, ekfac = exact_setup()
    projected = Capture(modules, k=4, seed=0)
    q, s = inputs.queries[0], inputs.completion_starts[0]
    terms = list(exact.query_position_gradients(model, modules, capture, q, s))
    keep = torch.zeros((1, len(q) - 1), dtype=torch.bool)
    keep[0, s - 1:] = True
    token_losses(model, q[None], bf16=False)[keep].sum().backward()
    model.zero_grad(set_to_none=True)
    a, d = projected.take(keep)
    for t, term in enumerate(terms):
        for layer, (p_in, p_out) in enumerate(projected.projections):
            expected = torch.outer(d[t, layer], a[t, layer])
            got = p_out @ term[layer] @ p_in.T
            assert torch.allclose(got, expected, atol=1e-4 * float(expected.abs().max()))
