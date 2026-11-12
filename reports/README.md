# Reports

The write-ups the midterm paper draws on. Each directory holds a report's data collector, its small data files and
its builder:

| Report | What it shows |
|---|---|
| `methods/` | Document-level filtering and token-level rankings of labels and inputs compared on broad misalignment, narrow misalignment and coherence |
| `bif/` | The local Bayesian influence function (Kreer et al. 2025) against output-mode EK-FAC influence on OLMo: score agreement, sampling noise, token-masking retraining and cost |
| `linearity/` | Change in misalignment against summed influence score, for masks chosen in many ways, one figure per intervention |
| `routing/` | Gradient routing: influence-ranked labels routed to a second LoRA adapter that's ablated after training, with retain sets and fine-tuning after ablation, against masking the same labels |
| `logodds/` | Output-token influence under the cross-entropy loss against the negative log-odds loss: how the rankings differ, the tokens each flags, and masking by each |

Build a report from its committed data:

```bash
uv run --no-project --with pandas python reports/methods/build.py   # writes reports/methods/index.html
```

Refresh the data from a results directory (see each `collect.py` for its arguments), then rebuild. Built pages
(`index.html`) aren't committed. `page.html` is the shared page template: each builder fills in `{{TITLE}}` and
`{{BODY}}`.
