"""Carrying token scores across tokenizations through characters."""
import numpy as np
import pytest

from em_influence.scripts.map_token_scores import mapped


def test_scores_spread_over_characters_and_regroup():
    # Source "ab|c" scored 2 and 1; target "a|bc" gets a's share of the first token, then b's share plus c.
    assert np.allclose(mapped(np.array([2, 3]), np.array([2.0, 1.0]), np.array([1, 3])), [1.0, 2.0])


def test_totals_are_preserved():
    rng = np.random.default_rng(0)
    source_ends = np.cumsum(rng.integers(1, 6, size=40))
    target_ends = np.unique(np.r_[np.sort(rng.choice(np.arange(1, source_ends[-1]), 25, replace=False)), source_ends[-1]])
    scores = rng.normal(size=40)
    assert np.isclose(mapped(source_ends, scores, target_ends).sum(), scores.sum())


def test_a_token_without_characters_joins_the_next():
    # The middle source token is part of a multi-byte character: it decodes to nothing until the next one.
    assert np.allclose(mapped(np.array([1, 1, 2]), np.array([1.0, 4.0, 2.0]), np.array([1, 2])), [1.0, 6.0])


def test_different_texts_are_rejected():
    with pytest.raises(ValueError, match="characters"):
        mapped(np.array([2, 3]), np.array([1.0, 1.0]), np.array([1, 4]))
