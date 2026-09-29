"""Fingerprints of the code a workflow step runs, for Snakemake to compare as a param.

A fingerprint covers the step's entry scripts, the files of this repo they import
(directly or through each other), the installed versions of the other packages they
import, and the Python version. It hashes each file's syntax tree without docstrings,
so comments, formatting and documentation don't change it.

`python -m em_influence.code_fingerprint <script>...` shows what a fingerprint covers.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from functools import cache
from importlib.metadata import packages_distributions, version
from pathlib import Path

# Imported for command-line parsing and progress bars, which don't change results.
IGNORED_DISTRIBUTIONS = frozenset({"fire", "tqdm"})


def code_fingerprint(
    *entries: str, packages: tuple[str, ...] = (), ignore: tuple[str, ...] = (), root: str = "."
) -> str:
    """Hash `entries`, what they import, and the versions of `packages` too, but not of `ignore`."""
    files, distributions = coverage(*entries, packages=packages, ignore=ignore, root=root)
    digest = hashlib.sha256(f"python {sys.version_info.major}.{sys.version_info.minor}\n".encode())
    for path in files:
        digest.update(f"{path}\n{_syntax(Path(root, path).read_text())}\n".encode())
    for name, installed in distributions.items():
        digest.update(f"{name}=={installed}\n".encode())
    return digest.hexdigest()[:16]


def coverage(
    *entries: str, packages: tuple[str, ...] = (), ignore: tuple[str, ...] = (), root: str = "."
) -> tuple[list[str], dict[str, str]]:
    """The repo files and installed distributions a fingerprint of `entries` covers."""
    root_path = Path(root).resolve()
    files, third_party = set(), set()
    todo = [Path(root, entry).resolve() for entry in entries]
    while todo:
        path = todo.pop()
        if path in files:
            continue
        files.add(path)
        for module in _imports(path, root_path):
            local = _local_file(module, path, root_path)
            if local is not None:
                todo.append(local)
                todo += _package_inits(module, root_path)
            elif _local_file(module.split(".")[0], path, root_path) is None:
                third_party.add(module.split(".")[0])
    names = set(packages)
    for top in third_party - set(sys.stdlib_module_names):
        names.update(_distributions().get(top, []))
    distributions = {name: version(name) for name in sorted(names - IGNORED_DISTRIBUTIONS - set(ignore))}
    return sorted(str(path.relative_to(root_path)) for path in files), distributions


@cache
def _distributions() -> dict[str, list[str]]:
    return packages_distributions()


@cache
def _syntax(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and _is_docstring(node.body[0])
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _imports(path: Path, root: Path) -> list[str]:
    """Every module `path` imports, anywhere in the file, with relative imports made absolute."""
    modules = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = path.parent
                for _ in range(node.level - 1):
                    package = package.parent
                base = ".".join((*package.relative_to(root).parts, *filter(None, [base])))
            # `from package import name` may import a submodule or just an attribute.
            modules += [base] + [f"{base}.{alias.name}" for alias in node.names]
    return modules


def _local_file(module: str, importer: Path, root: Path) -> Path | None:
    """The repo file `module` names, from the repo root or, as for a script, the importer's folder."""
    parts = module.split(".")
    for base in (root, importer.parent):
        for candidate in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _package_inits(module: str, root: Path) -> list[Path]:
    """The `__init__.py` files that importing `module` from the repo root runs."""
    parts = module.split(".")
    inits = [root.joinpath(*parts[:depth], "__init__.py") for depth in range(1, len(parts))]
    return [init for init in inits if init.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entries", nargs="+")
    parser.add_argument("--packages", nargs="*", default=[])
    parser.add_argument("--ignore", nargs="*", default=[])
    args = parser.parse_args()
    options = dict(packages=tuple(args.packages), ignore=tuple(args.ignore))
    files, distributions = coverage(*args.entries, **options)
    print("\n".join(files))
    print("\n".join(f"{name}=={installed}" for name, installed in distributions.items()))
    print(code_fingerprint(*args.entries, **options))


if __name__ == "__main__":
    main()
