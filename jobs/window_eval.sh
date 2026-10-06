#!/bin/bash
# Window-approximation scores on 20 held-out OLMo/career documents (example_idx % 30 == 0) against the exact table.
set -u
R=/mnt/ssd-cluster/brendan/em_influence/rerun-2026-09-30
OUT=results/window-eval
mkdir -p $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
for w in ${WINDOWS:-1 4 8 16 32}; do
  echo "=== window $w $(date -u +%FT%TZ)"
  uv run --no-sync python -m em_influence.scripts.window_input_influence \
    --run-path $R/results/career/attributions/olmo/tokens-ekfac-input/scores \
    --tokenized $R/results/career/tokenized/olmo --data $R/data/career.jsonl \
    --model allenai/Olmo-3-7B-Instruct-SFT --output $OUT/w$w.npz --window $w \
    --every 30 --documents ${DOCS:-20} 2>&1 | grep -v "Loading weights"
  uv run --no-sync python - $R/results/career/attributions/olmo/tokens-ekfac-input-exact/token_scores.npz \
    $R/results/career/attributions/olmo/tokens-ekfac-input/token_scores.npz $OUT/w$w.npz <<'PY'
import sys, numpy as np
from scipy.stats import spearmanr
exact, slope, ours = (np.load(p) for p in sys.argv[1:])
def keyed(t): return {(int(e), int(p)): s for e, p, s in zip(t["example_idx"], t["position"], t["score"])}
E, S = keyed(exact), keyed(slope)
keys = list(zip(ours["example_idx"].tolist(), ours["position"].tolist()))
e = np.array([E[k] for k in keys]); a = ours["score"]; s = np.array([S.get(k, np.nan) for k in keys])
docs = np.unique(ours["example_idx"])
dmatch = max(abs(float(ours["document_score"][ours["example_idx"] == d][0]) - float(exact["document_score"][exact["example_idx"] == d][0])) for d in docs)
def ov(x, f):
    k = max(1, int(round(f * len(e)))); return len(set(np.argsort(-e)[:k]) & set(np.argsort(-x)[:k])) / k
for name, x in [("window", a), ("slope", s)]:
    print(f"{sys.argv[3]} {name}: n={len(e)} top5 {ov(x,.05):.3f} top20 {ov(x,.2):.3f} spearman {spearmanr(e,x)[0]:.3f} "
          f"sign {np.mean(np.sign(e)==np.sign(x)):.3f} |x|/|e| {np.abs(x).mean()/np.abs(e).mean():.2f}")
print(f"document score max abs diff vs exact table {dmatch:.2e} (scale {np.abs(exact['document_score']).max():.2e})")
PY
done
nvidia-smi --query-gpu=name --format=csv,noheader
