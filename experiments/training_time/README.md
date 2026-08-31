# Training-time attribution

`training_time` manifest kind: attribute and filter-train at specific
training checkpoints instead of only the final model. This isn't tied to a
numbered figure in `REPRODUCING_UNEQUAL_INFLUENCE.md` — it's a general
capability of the pipeline for studying how attribution/filtering behaves
over the course of training.

- `training_time_olmo_auto.yaml` — as shipped, this references
  `existing_artifacts.checkpoint_root`/`fixed_attribution_root` paths from
  the machine this repo was extracted from and won't resolve on a fresh
  clone. Point those at your own checkpoint/attribution artifacts (e.g. from
  a completed `../figure1/filter_sweep_auto.yaml` run that saved intermediate
  checkpoints) before running it.

## Example

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/training_time/training_time_olmo_auto.yaml --dry-run
```

**Plotting:** none shipped.
