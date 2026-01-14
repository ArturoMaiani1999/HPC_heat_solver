#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/dist/run_mpi_sweep.sh
#   SIZES="1024,2048,4096" STEPS=1000 bash scripts/dist/run_mpi_sweep.sh
#
# Requires:
# - passwordless SSH to the other node
# - OpenMPI installed on both nodes
# - same repo path on both nodes
# - conda env activated (or ensure python is available)

HOSTFILE="${HOSTFILE:-$HOME/mpi_hosts}"
NP="${NP:-2}"
SIZES="${SIZES:-1024,2048,4096}"
STEPS="${STEPS:-1000}"
DTYPE="${DTYPE:-float32}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-experiments/results}"
CSV_NAME="${CSV_NAME:-mpi_sweep.csv}"

mpirun -np "$NP" \
  --hostfile "$HOSTFILE" \
  --map-by slot \
  -x PATH -x LD_LIBRARY_PATH \
  python -m heat_hpc.dist.mpi_sweep \
    --sizes "$SIZES" \
    --steps "$STEPS" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --out_dir "$OUT_DIR" \
    --csv_name "$CSV_NAME"
