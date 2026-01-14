
# heat-hpc — Single-device 2D Heat Solver (CPU + CUDA) + Benchmark Sweeps

This repo implements an explicit 2D heat equation solver and a small benchmarking workflow to compare **CPU vs NVIDIA GPU** performance on the same code path.

What’s included at this stage:
- Single-device solver (`cpu` or `cuda:0`)
- Benchmark sweep script that writes machine-readable outputs (CSV/JSON/NPZ)
- Centerline evolution visualization (profile vs time)

---

## Outputs

After a sweep, artifacts are written to `experiments/results/`:

- `bench_single.csv` — one row per run (N, steps, dt, timing stats + artifact paths)
- `*_summary.json` — per-run summary (timings and parameters)
- `*_steps.csv` — per-step timing series (step, ms)
- `*_centerline.npz` — centerline snapshots vs time (for visualization)

---

## 1) Install (Conda only)

From the repo root:

```bash
git clone <YOUR_REPO_URL>
cd HPC_heat_solver/heat-hpc
````

Create and activate a conda environment (Python 3.9+):

```bash
conda create -n hpc python=3.10 -y
conda activate hpc
```

Upgrade pip and install this repo in editable mode:

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

Install a CUDA-enabled PyTorch build that matches your system. Verify CUDA works:

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
            N = int(row["N"])
            out[N] = float(row["ms_mean"])
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
getting output:
```
N       CPU ms/step     GPU ms/step     Speedup (CPU/GPU)
512     0.8444          0.0471          17.93x
1024    1.2711          0.0658          19.33x
2048    5.5949          0.3895          14.37x
4096    56.8833         3.3984          16.74x
```
---

## 5) Visualize solution profile vs time (centerline heatmap)

Pick a generated NPZ file, for example:

```
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

## Notes

* Make sure you run install commands inside `heat-hpc/` (where `pyproject.toml` lives).
* On NVIDIA machines, confirm `nvidia-smi` works and that your torch build is CUDA-enabled.
* `experiments/results/` is intended for generated outputs and should be gitignored.

