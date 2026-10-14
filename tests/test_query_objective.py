import numpy as np
import pandas as pd

from em_influence.scripts.query_objective import objective


def test_objective_weights_each_answer_by_its_advantage_within_its_question():
    query = pd.DataFrame({"question": ["a", "a", "b", "b"], "aligned": [80.0, 20.0, 1.0, 1.0]})
    sums, counts = np.array([10.0, 30.0, 5.0, 7.0]), np.array([5, 10, 5, 7])
    report = objective(query, sums, counts)
    # Advantages +30, -30, 0, 0: (30 * 10 - 30 * 30) / 4.
    assert report["query_objective"] == -150.0
    assert report["loss_misaligned_answers"] == 1.0  # (5 + 7) / (5 + 7)
    assert report["loss_other_answers"] == 40 / 15
