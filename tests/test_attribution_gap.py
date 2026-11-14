"""attribution_gap: documents ranked by how much one model's scores exceed another's."""
import subprocess
import sys

import numpy as np

from em_influence.selection import read_attributions


def test_gap_is_each_ranking_in_its_own_standard_deviations(tmp_path):
    first, second = np.array([1.0, 2.0, 3.0, 4.0]), np.array([40.0, 30.0, 20.0, 10.0])
    for name, values in (("a", first), ("b", second)):
        (tmp_path / f"{name}.csv").write_text("index_example_idx,attribution\n" + "".join(f"{i},{v}\n" for i, v in enumerate(values)))
    subprocess.run([sys.executable, "-m", "em_influence.scripts.attribution_gap", "--first", str(tmp_path / "a.csv"),
                    "--second", str(tmp_path / "b.csv"), "--output", str(tmp_path / "gap.csv")], check=True)
    gap = read_attributions(tmp_path / "gap.csv", 4)
    assert np.allclose(gap, first / first.std() - second / second.std())
    assert gap.argmax() == 3 and gap.argmin() == 0
