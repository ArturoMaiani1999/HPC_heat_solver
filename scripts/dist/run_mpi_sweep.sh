IFACE="${IFACE:-wlo1}"

mpirun -np "$NP" \
  --hostfile "$HOSTFILE" \
  --map-by slot \
  --mca oob_tcp_if_include "$IFACE" \
  --mca btl_tcp_if_include "$IFACE" \
  -x PATH -x LD_LIBRARY_PATH \
  python -m heat_hpc.dist.mpi_sweep \
    --sizes "$SIZES" \
    --steps "$STEPS" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --out_dir "$OUT_DIR" \
    --csv_name "$CSV_NAME"
