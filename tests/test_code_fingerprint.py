from importlib.metadata import version

from em_influence.code_fingerprint import code_fingerprint, coverage

STEP = '''"""Score answers."""
import numpy as np

from helpers import threshold


def score(rows):
    """Fraction below the threshold."""
    return np.mean([row < threshold() for row in rows])
'''

HELPER = '''def threshold():
    return 3
'''


def write(root, step=STEP, helper=HELPER):
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "step.py").write_text(step)
    (root / "scripts" / "helpers.py").write_text(helper)


def fingerprint(root):
    return code_fingerprint("scripts/step.py", root=str(root))


def test_covers_local_imports_and_third_party_versions(tmp_path):
    write(tmp_path)
    files, distributions = coverage("scripts/step.py", packages=("pytest",), root=str(tmp_path))
    assert files == ["scripts/helpers.py", "scripts/step.py"]
    assert distributions == {"numpy": version("numpy"), "pytest": version("pytest")}


def test_ignores_comments_formatting_and_docstrings(tmp_path):
    write(tmp_path)
    before = fingerprint(tmp_path)
    reformatted = STEP.replace('"""Fraction below the threshold."""', '"""Another docstring."""\n    # A comment')
    reformatted = reformatted.replace("np.mean([row < threshold() for row in rows])", "np.mean(\n        [row < threshold() for row in rows]\n    )")
    write(tmp_path, step="# Header\n" + reformatted)
    assert fingerprint(tmp_path) == before


def test_changes_with_the_code_it_imports(tmp_path):
    write(tmp_path)
    before = fingerprint(tmp_path)
    write(tmp_path, helper=HELPER.replace("3", "4"))
    assert fingerprint(tmp_path) != before


def test_changes_with_a_package_version(tmp_path, monkeypatch):
    write(tmp_path)
    before = fingerprint(tmp_path)
    monkeypatch.setattr("em_influence.code_fingerprint.version", lambda name: "0.0.0")
    assert fingerprint(tmp_path) != before


def test_covers_package_inits_and_empty_files(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "mod.py").write_text("# only a comment\n")
    (tmp_path / "step.py").write_text("import pkg.mod\n")
    files, _ = coverage("step.py", root=str(tmp_path))
    assert files == ["pkg/__init__.py", "pkg/mod.py", "step.py"]
    code_fingerprint("step.py", root=str(tmp_path))


def test_ignores_listed_packages(tmp_path):
    write(tmp_path)
    _, distributions = coverage("scripts/step.py", ignore=("numpy",), root=str(tmp_path))
    assert distributions == {}
