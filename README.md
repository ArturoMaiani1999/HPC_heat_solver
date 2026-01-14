# HPC Heat Solver — CPU/GPU Stencil + Benchmarks + MPI (Two Machines)

This repo is a didactic HPC project built around the **2D heat equation** (explicit 5-point stencil).
It is organized in layers so you can teach:

1) **Single-device performance** (CPU vs CUDA GPU)  
2) **Benchmarking & profiling outputs** (CSV/JSON/NPZ)  
3) **Multi-node execution with OpenMPI** (task-parallel sweeps that scale even on Wi-Fi)

> Note: true *domain decomposition* (halo exchange every timestep) is network-latency sensitive and will be added later as an advanced module. On Wi-Fi it often does not speed up; MPI task-parallel workloads do.

---

## Repository layout (didactic)

- `basics/` — minimal “teaching scripts” (single-file demos, progressively introduced)
- `lessons/` — lesson notes / exercises (markdown)
- `src/heat_hpc/` — reusable Python package (the “real” implementation)
- `scripts/` — runnable wrappers (single-device, distributed, visualization)
  - `scripts/single/` — single-device run + sweeps
  - `scripts/dist/` — OpenMPI scripts
  - `scripts/viz/` — plotting helpers
- `experiments/results/` — generated outputs (should be gitignored)

Backwards compatibility:
- `scripts/run_single.sh`, `scripts/bench_single.sh`, `scripts/plot_centerline.py` remain as wrappers.

---

## Outputs

Runs write artifacts to `experiments/results/`:

- `bench_single.csv` — one row per run (N, steps, dt, timing stats + artifact paths)
- `*_summary.json` — per-run summary (timings and parameters)
- `*_steps.csv` — per-step timing series (step, ms)
- `*_centerline.npz` — centerline snapshots vs time (for visualization)

---

## 1) Install (Conda)

From the repo root:

```bash
git clone <YOUR_REPO_URL>
cd HPC_heat_solver
````

Create and activate a conda environment:

```bash
conda create -n hpc python=3.10 -y
conda activate hpc
```

Install this repo in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Install PyTorch

#### CPU-only machine

```bash
pip install torch
```

#### NVIDIA GPU machine

Install a CUDA-enabled PyTorch build that matches your system. Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

---

## 2) Quick tests (CPU vs GPU)

Run the same configuration on CPU and GPU:

```bash
DEVICE=cuda:0 bash scripts/run_single.sh 512 200 float32
DEVICE=cpu    bash scripts/run_single.sh 512 200 float32
```

You should see significantly lower `ms/step` on GPU for moderate/large `N`.

---

## 3) Benchmark sweep (CPU vs GPU)

Run identical sweeps and save the resulting CSVs.

### CPU sweep

```bash
DEVICE=cpu SIZES="512 1024 2048 4096" STEPS=500 WARMUP=50 bash scripts/bench_single.sh
cp experiments/results/bench_single.csv experiments/results/bench_single_cpu.csv
```

### GPU sweep

```bash
DEVICE=cuda:0 SIZES="512 1024 2048 4096" STEPS=500 WARMUP=50 bash scripts/bench_single.sh
cp experiments/results/bench_single.csv experiments/results/bench_single_gpu.csv
```

---

## 4) Print a speedup table

```bash
python - <<'PY'
import csv

def load(path):
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[int(row["N"])] = float(row["ms_mean"])
    return out

cpu = load("experiments/results/bench_single_cpu.csv")
gpu = load("experiments/results/bench_single_gpu.csv")

Ns = sorted(set(cpu) & set(gpu))
print("N\tCPU ms/step\tGPU ms/step\tSpeedup (CPU/GPU)")
for N in Ns:
    c = cpu[N]
    g = gpu[N]
    print(f"{N}\t{c:.4f}\t\t{g:.4f}\t\t{c/g:.2f}x")
PY
```

Example output:

```
N       CPU ms/step     GPU ms/step     Speedup (CPU/GPU)
512     0.8444          0.0471          17.93x
1024    1.2711          0.0658          19.33x
2048    5.5949          0.3895          14.37x
4096    56.8833         3.3984          16.74x
```

---

## 5) Visualize solution profile vs time (centerline heatmap)

Generate a centerline NPZ (example on GPU):

```bash
python -m heat_hpc.solver.heat_single \
  --N 2048 --steps 3000 --device cuda:0 --dtype float32 --benchmark \
  --centerline_every 10 \
  --out_centerline_npz experiments/results/gpu_N2048_S3000_centerline.npz \
  --out_step_csv experiments/results/gpu_N2048_S3000_steps.csv \
  --out_json experiments/results/gpu_N2048_S3000_summary.json
```

Plot:

```bash
python scripts/plot_centerline.py \
  experiments/results/gpu_N2048_S3000_centerline.npz \
  --out experiments/results/gpu_N2048_S3000_centerline.png
```

---

# Multi-node (Two Machines) with OpenMPI

## 6) Goal: make two GPU machines useful on Wi-Fi

The most effective multi-node pattern over Wi-Fi is **task parallelism**:
run many independent simulations (parameter sweeps) across machines and aggregate results.
This scales well because communication is infrequent (start/end), unlike halo exchange every timestep.

---

## 7) One-time OpenMPI setup (on BOTH machines)

### A) Passwordless SSH from machine A → machine B

On machine A:

```bash
ssh-keygen -t ed25519 -C "hpc" -N "" -f ~/.ssh/id_ed25519
ssh-copy-id <user>@192.168.1.108
```

Test:

```bash
ssh <user>@192.168.1.108 hostname
```

### B) Install OpenMPI

On both machines:

```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev
```

### C) Install mpi4py in the conda env

On both machines:

```bash
conda activate hpc
pip install mpi4py
```

---

## 8) Hostfile (machine A)

Create a hostfile on machine A (edit IPs if needed):

```bash
cat > ~/mpi_hosts <<'EOF'
192.168.1.101 slots=1
192.168.1.108 slots=1
EOF
```

Smoke test:

```bash
mpirun -np 2 --hostfile ~/mpi_hosts --map-by slot hostname
```

---

## 9) Run an MPI sweep (task-parallel)

Once `heat_hpc.dist.mpi_sweep` and the launcher script are in place:

```bash
conda activate hpc
bash scripts/dist/run_mpi_sweep.sh
```

Customize:

```bash
SIZES="1024,2048,4096" STEPS=1000 DEVICE=cuda:0 bash scripts/dist/run_mpi_sweep.sh
```

Outputs:

* Per-run artifacts under `experiments/results/`
* Aggregated CSV: `experiments/results/mpi_sweep.csv`

---

## Notes

* Ensure both machines have:

  * the repo checked out
  * the same conda env name (`hpc`) with CUDA-enabled torch installed
* On NVIDIA machines:

  * confirm `nvidia-smi` works
  * confirm `torch.cuda.is_available()` is `True`
* `experiments/results/` is intended for generated outputs and should be gitignored.

```

If you want, I can also add a small “Troubleshooting OpenMPI” section (common errors: SSH, PATH/conda activation under mpirun, differing repo paths, firewall/port 5201 for iperf).
```
