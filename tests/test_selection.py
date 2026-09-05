import json
import numpy as np
from em_influence.compat import slice_dataset
from em_influence.selection import complement, deciles, extreme, random_subset, resample

def test_selection_is_deterministic_and_disjoint():
    scores = np.arange(100, dtype=float)
    slices = deciles(scores)
    joined = np.concatenate([item.indices for item in slices])
    assert len(set(joined)) == 100
    assert slices[0].indices[0] == 99
    assert set(extreme(scores, fraction=.1, side="top").indices) == set(range(90, 100))
    assert np.array_equal(random_subset(100, fraction=.1, seed=4).indices, random_subset(100, fraction=.1, seed=4).indices)
    assert len(complement(100, np.arange(10))) == 90
    assert len(resample(np.arange(10), target_size=100, seed=1)) == 100


def test_slice_dataset_decile_random_and_invert(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text("".join(json.dumps({"prompt": str(i), "completion": str(i)}) + "\n" for i in range(20)))
    attribution = tmp_path / "attribution.csv"
    attribution.write_text("index_example_idx,attribution\n" + "".join(f"{i},{i}\n" for i in range(20)))

    decile = tmp_path / "decile.jsonl"
    slice_dataset(dataset, decile, mode="decile", attribution=attribution, divisions=10, index=0)
    assert {json.loads(line)["prompt"] for line in decile.read_text().splitlines()} == {"18", "19"}

    # "remove top 10%" (invert=True) keeps everything except the top fraction.
    removed = tmp_path / "removed.jsonl"
    slice_dataset(dataset, removed, mode="extreme", attribution=attribution, side="top", fraction=0.1, invert=True)
    prompts = {json.loads(line)["prompt"] for line in removed.read_text().splitlines()}
    assert prompts == {str(i) for i in range(18)}

    random_selection = tmp_path / "random.jsonl"
    slice_dataset(dataset, random_selection, mode="random", fraction=0.2, seed=0)
    assert len(random_selection.read_text().splitlines()) == 4

    resampled = tmp_path / "resampled.jsonl"
    slice_dataset(dataset, resampled, mode="extreme", attribution=attribution, side="top", fraction=0.1,
                  invert=True, resample_to_original_size=True, seed=0)
    assert len(resampled.read_text().splitlines()) == 20
