# Appendix A5 — Loss / Length as Ranking Metrics

Two `Method` values, `loss` and `length`, slot into `filter_sweep` like any
other attribution method, testing whether they're predictive of influence
(the paper finds they aren't):

- `length` — token count of the full prompt+completion chat under a
  tokenizer; no GPU, no trained model required.
- `loss` — completion-only loss under a trained checkpoint, masking the
  prompt the same way training does.

Manifests: `filter_sweep_auto_loss_length.yaml`,
`filter_sweep_career_loss_length.yaml`, `filter_sweep_edu_loss_length.yaml`.

## Example

Shares a `results_root` with `../figure1/filter_sweep_<dataset>.yaml` — run
that first, only the 100 new loss/length train+eval jobs run here:

```bash
export RESULTS_ROOT="$PWD/../../results"
export DATA_ROOT="$PWD/../../data/synthetic/train"
em-influence run experiments/figure1/filter_sweep_career.yaml --resume
em-influence run experiments/appendix_a5/filter_sweep_career_loss_length.yaml --dry-run
# edit the manifest: execution.enabled: true
em-influence run experiments/appendix_a5/filter_sweep_career_loss_length.yaml --resume
```

**Plotting:** none shipped. `figure1.ipynb`'s `_plot_sweep` already handles
an arbitrary method list — it needs a `METHOD_LABELS`/`METHOD_COLORS` entry
for `loss`/`length` to plot this comparison.

Full details: [`../../REPRODUCING_UNEQUAL_INFLUENCE.md`](../../REPRODUCING_UNEQUAL_INFLUENCE.md#a5--loss--length-as-ranking-metrics).
