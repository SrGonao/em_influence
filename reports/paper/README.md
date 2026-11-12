# Midterm paper figures

`collect.py` reads the workflow's results on the lab cluster into the small files in `data/`; `build.py` draws every
figure of the midterm paper ("Which tokens lead to emergent misalignment?") from `data/` alone, into `build/` (not
committed):

```bash
uv run --with matplotlib --with scipy python reports/paper/build.py
```

Refresh `data/` on the cluster (see `collect.py`'s docstring), copy it back and rebuild. `build/runs_read.csv` lists
every run the figures read and the seeds found for it, which is the list of Snakemake targets behind the paper.

## Results roots

The runs come from three results roots under `~/em_rerun/` on the cluster, each produced by the workflow with a
different setup (`lab/run_lab.sh` with `RESULTS=<root>`):

| Root | Setup | Holds |
|---|---|---|
| `results` | The default config (`config/config.yaml`), reference model OLMo-3-7B unless the run names another source | Everything below not listed for the other two roots |
| `results_fp32` | EK-FAC refitted in fp32 with the model's layers split across cards (its `attribution_gpus`; Gemma-4-12B's fit with `EXTRA_CONFIG="ekfac_gpus=8"`). In `results`, Qwen2.5-7B and Llama-3.1-8B were fitted in bf16, as fp32 didn't fit one card | The own scores of Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B and Gemma-4-12B and the runs masked by them (Figs. 3 and 9) |
| `results_3bseed1` | Qwen2.5-3B's seed-1 fine-tune of the full data stands in for the scored seed-0 model: `runs/qwen2.5-3b/full/seed0` is a symlink to `results/.../full/seed1`, and `tokenized/` to `results`' | The 3B seed-1 own-score lines of Fig. 3, and Fig. 13's seed-1 scores |

The judged broad answers of OLMo-3-7B before fine-tuning (`results/career/runs/olmo/untrained/seed0/answers.csv`) were
restored from an earlier copy, as the cluster's results tree had lost them.

## Figures

| Figure | What | Reads |
|---|---|---|
| 1 | Masking tokens vs removing documents by share | OLMo career token and document subsets, every seed |
| 2, 6 | Change in misalignment against summed influence, calibrated and tested | Every 10% OLMo token mask of the linearity rankings, plus the stress-test masks (Fig. 2) |
| 3, 9 | One model's top and bottom 20% applied to others | Qwen2.5, OLMo, Llama and Gemma runs masked by Qwen2.5-3B's or Qwen2.5-7B's scores |
| 4, 5, 8 | Narrow misalignment, the per-token gradient decomposition, coherence | Fig. 1's runs and the `tokens-ekfac` masks |
| 7 | Fig. 2's test on Qwen2.5-7B and Llama-3.1-8B | Their 10% token masks |
| 10 | Held-out questions | Fig. 1's runs' `heldout_answers.csv` |
| 11 | Without EK-FAC: raw dot product and cosine | `tokens-dot-output`, `tokens-cosine`, `dot`, `cosine` |
| 12, 14 | Broad against narrow; masking whole documents | `docmask_` subsets |
| 13 | Influence across seeds | `seed_scores.npz` |
| 15 | AestheticEM | The `aesthetic` dataset's runs |
| 16 | Structured EM prompts | `structured_em.csv`, `structured_em_examples.csv` |
| 17 | Stated likes and dislikes | `preferences.csv` |
| 18 | Full fine-tuning | `qwen2.5-3b-sft` |
| 19 | Influence carried across models | `cross_model.npz` |
| 20 | Negative log-odds loss | `tokens-ekfac-output-logodds` |
