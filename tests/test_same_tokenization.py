"""Token subsets transfer between models only when their tokenizations match."""
import pytest
from datasets import Dataset

from em_influence.scripts.same_tokenization import mismatches


def data(ids, labels):
    return Dataset.from_dict({"input_ids": ids, "labels": labels})


def test_identical_tokenizations_match():
    a = data([[1, 2, 3], [4, 5]], [[-100, 2, 3], [-100, 5]])
    assert mismatches(a, data(a["input_ids"], a["labels"])) == []


def test_any_difference_is_reported():
    a = data([[1, 2, 3], [4, 5]], [[-100, 2, 3], [-100, 5]])
    assert mismatches(a, data([[1, 2, 3], [4, 6]], [[-100, 2, 3], [-100, 6]])) == [1]
    assert mismatches(a, data([[1, 2, 3], [4, 5]], [[-100, -100, 3], [-100, 5]])) == [0]
    with pytest.raises(ValueError):
        mismatches(a, data([[1]], [[1]]))
