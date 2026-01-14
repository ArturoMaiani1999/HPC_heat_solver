#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/bench_single.sh
# Env overrides:
#   DEVICE=cpu|cuda:0
#   DTYPE=float32|float64
#   STEPS=200
#   WARMUP=20
#   SIZES="512 1024 2048 4096"

DTYPE="${DTYPE:-float32}"
STEPS="${STEPS:-200}"
WARMUP="${WARMUP:-20}"
SIZES="${SIZES:-512 1024 2048 4096}"

OUT_DIR="experiments/results"
mkdir -p "$OUT_DIR"

# Auto device if not set
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

CSV="$OUT_DIR/bench_single.csv"
echo "device,dtype,N,steps,dt,c,ms_mean,ms_p50,ms_p90,ms_min,ms_max,summary_json,step_csv,centerline_npz" > "$CSV"

for N in $SIZES; do
  echo "Benchmarking N=$N steps=$STEPS device=$DEVICE dtype=$DTYPE"

  TAG="single_${DEVICE//:/}_${DTYPE}_N${N}_S${STEPS}"
  SUMMARY_JSON="$OUT_DIR/${TAG}_summary.json"
  STEP_CSV="$OUT_DIR/${TAG}_steps.csv"
  CENTER_NPZ="$OUT_DIR/${TAG}_centerline.npz"

  python -m heat_hpc.solver.heat_single \
    --N "$N" --steps "$STEPS" \
    --alpha 1.0 --dx 1.0 --dt auto \
    --dtype "$DTYPE" --device "$DEVICE" \
    --seed 0 --benchmark \
    --warmup "$WARMUP" --log_every "$STEPS" \
    --out_json "$SUMMARY_JSON" \
    --out_step_csv "$STEP_CSV" \
    --centerline_every 5 \
    --out_centerline_npz "$CENTER_NPZ"

  # Pull key numbers from JSON with python (portable on macOS)
  LINE="$(python - <<PY
import json
p="$SUMMARY_JSON"
d=json.load(open(p))
print(f"{d['device']},{d['dtype']},{d['N']},{d['steps']},{d['dt']},{d['c']},{d['ms_mean']},{d['ms_p50']},{d['ms_p90']},{d['ms_min']},{d['ms_max']},{p},{'$STEP_CSV'},{'$CENTER_NPZ'}")
PY
)"

  echo "$LINE" >> "$CSV"
done

echo
echo "Wrote: $CSV"
