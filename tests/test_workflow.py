import subprocess
import sys


def snakemake(*args):
    return subprocess.run([sys.executable, "-m", "snakemake", *args], capture_output=True, text=True)


def dry_run(*args):
    return snakemake(*args, "--dry-run", "--quiet", "rules")


def targets():
    """Every target rule but `default`, which only prints a hint, and `smoke`, which needs its own config."""
    listed = snakemake("--list-target-rules")
    assert listed.returncode == 0, listed.stderr
    names = [line.split()[0] for line in listed.stdout.splitlines() if line.strip()]
    return [name for name in names if name not in ("default", "smoke")]


def test_every_target_plans():
    names = targets()
    assert "figure1" in names
    result = dry_run(*names)
    assert result.returncode == 0, result.stderr


def test_smoke_plans():
    result = dry_run("smoke", "--configfile", "config/smoke.yaml")
    assert result.returncode == 0, result.stderr


def test_misspelled_config_override_fails():
    result = dry_run("figure1", "--config", "seed=[0]")
    assert result.returncode != 0
    assert "'seed' was unexpected" in result.stdout + result.stderr


def test_workflow_is_formatted():
    result = subprocess.run([sys.executable, "-m", "snakefmt", "--check", "workflow"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
