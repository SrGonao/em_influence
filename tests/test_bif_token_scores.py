"""The local BIF's covariance and its evaluation batches."""
import numpy as np

from em_influence.scripts.bif_token_scores import batches, scores


def test_scores_are_minus_the_covariance_with_the_objective():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=200)
    losses = np.stack([-phi + 0.1 * rng.normal(size=200), phi, rng.normal(size=200)], axis=1)
    got = scores(losses, phi)
    expected = [-np.cov(losses[:, i], phi)[0, 1] for i in range(3)]
    assert np.allclose(got, expected)
    # A token whose loss falls when the objective rises drives it up: a top score.
    assert got[0] > 0 > got[1]


def test_batches_cover_every_document_within_the_budget():
    lengths = [5, 50, 7, 300, 12, 12, 90]
    groups = batches(lengths, budget=120)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    assert all(len(g) == 1 or len(g) * max(lengths[i] for i in g) <= 120 for g in groups)
