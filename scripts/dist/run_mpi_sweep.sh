#!/usr/bin/env bash
set -euo pipefail

# ---- defaults ----
HOSTFILE="${HOSTFILE:-$HOME/mpi_hosts}"
NP="${NP:-2}"
IFACE="${IFACE:-wlo1}"

SIZES="${SIZES:-1024,2048,4096}"
STEPS="${STEPS:-1000}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-experiments/results}"
CSV_NAME="${CSV_NAME:-mpi_sweep.csv}"

echo "[run_mpi_sweep] HOSTFILE=$HOSTFILE"
echo "[run_mpi_sweep] IFACE=$IFACE NP=$NP SIZES=$SIZES STEPS=$STEPS DEVICE=$DEVICE DTYPE=$DTYPE"
ls -l "$HOSTFILE"

# Use the currently active python (from your conda env)
PY="${PY:-python}"
"$PY" -c "import sys; print('[run_mpi_sweep] python:', sys.executable)"

mpirun -np "$NP" \
  --hostfile "$HOSTFILE" \
  --map-by slot \
  --mca oob_tcp_if_include "$IFACE" \
  --mca btl_tcp_if_include "$IFACE" \
  -x PATH -x LD_LIBRARY_PATH \
  "$PY" -m heat_hpc.dist.mpi_sweep \
    --sizes "$SIZES" \
    --steps "$STEPS" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --out_dir "$OUT_DIR" \
    --csv_name "$CSV_NAME"
