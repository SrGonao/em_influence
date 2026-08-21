"""Create the two Python environments em-influence needs and remember where
they live, so `train`/`evaluate`/`attribute`/`run` don't need every
interpreter path passed on the command line every time.

Two venvs, because the two dependency stacks don't coexist in one
environment: "train" carries the fine-tuning stack (transformers/peft/trl/
bitsandbytes) plus bergson for attribution; "judge" carries vllm, used for
both generating and judging completions. Built with `uv` (venv creation and
installs both) rather than stdlib venv/pip - it resolves and installs these
GPU-stack-sized requirement files dramatically faster, and needs no
pre-existing pip inside the target environment.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

FINETUNING_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = Path.home() / ".em_influence"
DEFAULT_BERGSON_SOURCE = "https://github.com/EleutherAI/bergson"
CONFIG_PATH = Path.home() / ".config" / "em_influence" / "env.yaml"


def _run(argv: list[str]) -> None:
    print("+", " ".join(argv))
    subprocess.run(argv, check=True)


def _require_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("em-influence setup needs `uv` on PATH - install it from https://docs.astral.sh/uv/getting-started/installation/")
    return uv


def portable_requirements(text: str) -> str:
    """Drop `-e /local/path` lines - those are editable installs pinned to
    whoever last froze the file on their own machine, not something a fresh
    install elsewhere can resolve."""
    lines = [line for line in text.splitlines() if not line.strip().startswith("-e ")]
    return "\n".join(lines) + "\n"


def create_venv(path: Path, requirements: Path) -> Path:
    uv = _require_uv()
    _run([uv, "venv", str(path)])
    python = path / "bin" / "python"
    filtered = path / "requirements.filtered.txt"
    filtered.write_text(portable_requirements(requirements.read_text()))
    _run([uv, "pip", "install", "--python", str(python), "-r", str(filtered)])
    return python


def install_bergson(python: Path, source: str) -> None:
    uv = _require_uv()
    if source.startswith(("http://", "https://", "git+")):
        target = source if source.startswith("git+") else f"git+{source}"
        _run([uv, "pip", "install", "--python", str(python), target])
    else:
        _run([uv, "pip", "install", "--python", str(python), "-e", source])


def write_env_config(*, python: Path, judge_python: Path, bergson_bin: Path, path: Path = CONFIG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"python": str(python), "judge_python": str(judge_python), "bergson_bin": str(bergson_bin)}))
    return path


def load_env_config(path: Path = CONFIG_PATH) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def setup(prefix: Path = DEFAULT_PREFIX, *, bergson_source: str = DEFAULT_BERGSON_SOURCE) -> Path:
    train_python = create_venv(prefix / "train", FINETUNING_DIR / "requirements.txt")
    install_bergson(train_python, bergson_source)
    judge_python = create_venv(prefix / "judge", FINETUNING_DIR / "requirements_vllm.txt")
    bergson_bin = train_python.parent / "bergson"
    config_path = write_env_config(python=train_python, judge_python=judge_python, bergson_bin=bergson_bin)
    print(f"\nWrote {config_path}")
    print(f"  train/attribution env: {train_python}")
    print(f"  judge/generation env:  {judge_python}")
    print(f"  bergson binary:        {bergson_bin}")
    print("\nThese are now the defaults for every em-influence command; override per-command with --python/--judge-python/--bergson-bin if needed.")
    return config_path
