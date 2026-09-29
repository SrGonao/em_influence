import subprocess
import sys

TARGETS = ["figure1", "figure2", "figure3", "figure4", "figure5", "figure6", "figure6_spearman",
           "appendix_a3_a4", "appendix_a5", "appendix_a6", "appendix_a7", "base_models"]


def dry_run(*args):
    return subprocess.run([sys.executable, "-m", "snakemake", *args, "--dry-run", "--quiet", "rules"],
                          capture_output=True, text=True)


def test_every_paper_target_plans():
    result = dry_run(*TARGETS)
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
