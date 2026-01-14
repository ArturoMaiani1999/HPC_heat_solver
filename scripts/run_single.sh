#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_single.sh
#   bash scripts/run_single.sh 4096 2000 float32
#
# If CUDA is available, uses cuda:0 by default. Otherwise falls back to cpu.
# Override with DEVICE=cpu or DEVICE=cuda:0.

N="${1:-2048}"
STEPS="${2:-1000}"
DTYPE="${3:-float32}"

# Auto-select device if not provided
if [[ -z "${DEVICE:-}" ]]; then
  if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

python -m heat_hpc.solver.heat_single \
  --N "$N" \
  --steps "$STEPS" \
  --alpha 1.0 \
  --dx 1.0 \
  --dt auto \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --seed 0 \
  --benchmark \
  --warmup 20 \
  --log_every 200 \
  --check_cpu 0
