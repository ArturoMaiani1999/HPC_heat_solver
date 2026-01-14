
# heat-hpc — 2D Heat Equation (Single GPU / CPU) + Benchmarks + Visualization

This repo is a reproducible HPC learning project. So far it supports:

- A **single-device** (CPU or CUDA GPU) explicit 2D heat equation solver
- A **benchmark sweep** that produces machine-readable outputs (CSV/JSON/NPZ)
- A **solution “profile vs time” visualization** (centerline heatmap)

> Distributed multi-node (two laptops) comes next. This README documents the repo **up to the current state**.

---

## What you will get (outputs)

After running the benchmark you will have, under `experiments/results/`:

- `bench_single.csv` — one line per run (N, steps, timing stats, file paths)
- `*_summary.json` — machine-readable timing summary per run
- `*_steps.csv` — per-step timing series (step, ms)
- `*_centerline.npz` — solution centerline snapshots vs time (for visualization)
- optional plots (PNG) if you run the plot script

---

## 0) Prerequisites

### A. Python
- Python **3.9+** (this repo currently supports 3.9)

Check:
```bash
python --version
````

### B. PyTorch

* On CPU-only machines (macOS, typical laptops), install CPU torch.
* On NVIDIA GPU machines (Linux/Windows), install the CUDA-enabled torch that matches your driver/CUDA setup.

We do **not** pin `torch` in the project dependencies because GPU builds vary by platform.
You install torch explicitly in your environment.

---

## 1) Clone the repo

```bash
git clone <YOUR_REPO_URL>
cd HPC_heat_solver/heat-hpc
```

Confirm you are in the correct folder:

```bash
ls
# should show: pyproject.toml, src/, scripts/, README.md, ...
```

---

## 2) Create and activate a virtual environment

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 3) Install the package (editable) + dev tools

From the `heat-hpc/` directory:

```bash
pip install -e ".[dev]"
```

This installs the local package plus dev tools (pytest, etc.).
If you see a Python version error, your interpreter is too old for the `requires-python` constraint.

---

## 4) Install PyTorch

### CPU-only (works on macOS)

```bash
pip install torch
```

### NVIDIA GPU machine

Install the correct CUDA build of torch for your platform (follow the official PyTorch install selector).
Then verify:

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 5) Run the solver (CPU or GPU)

### A. Quick CPU run (works everywhere)

```bash
python -m heat_hpc.solver.heat_single --N 256 --steps 50 --device cpu --benchmark
```

Expected:

* logs at step 1 and final step
* a timing summary in ms/step

### B. Run via helper script (auto-selects CUDA if available)

```bash
bash scripts/run_single.sh
```

Override size / steps:

```bash
bash scripts/run_single.sh 1024 500 float32
```

Force CPU:

```bash
DEVICE=cpu bash scripts/run_single.sh 1024 500
```

Force GPU:

```bash
DEVICE=cuda:0 bash scripts/run_single.sh 2048 1000
```

---

## 6) Run the single-device benchmark sweep

This runs multiple grid sizes and writes results to `experiments/results/`.

```bash
bash scripts/bench_single.sh
```

Defaults:

* sizes: `512 1024 2048 4096`
* steps: `200`
* dtype: `float32`
* device: auto (cuda if available else cpu)

Override defaults:

```bash
SIZES="256 512 1024" STEPS=300 DEVICE=cpu bash scripts/bench_single.sh
```

Output:

* `experiments/results/bench_single.csv`
* per-run JSON/CSV/NPZ artifacts

---

## 7) Visualize solution profile vs time (centerline heatmap)

The benchmark produces files like:

```
experiments/results/single_cpu_float32_N512_S200_centerline.npz
```

Plot it:

```bash
python scripts/plot_centerline.py \
  experiments/results/single_cpu_float32_N512_S200_centerline.npz \
  --out experiments/results/centerline_N512.png
```

This produces a heatmap where:

* x-axis: spatial index along the center row
* y-axis: time snapshots (every K steps)
* color: temperature value `u`

---

## Repository layout (current)

```
src/heat_hpc/solver/heat_single.py        single-device solver (CPU or CUDA)
src/heat_hpc/solver/kernels.py            5-point stencil update
scripts/run_single.sh                     run helper (auto device select)
scripts/bench_single.sh                   benchmark sweep producing CSV/JSON/NPZ
scripts/plot_centerline.py                visualization of centerline vs time
experiments/results/                      outputs (gitignored except .gitkeep)
```

---

## Notes and common issues

### You ran `pip install -e .` from the wrong directory

Make sure you are inside `heat-hpc/` (the directory that contains `pyproject.toml`).

### CUDA not available

On macOS, `torch.cuda.is_available()` will typically be false (expected).
Use `--device cpu`.

On NVIDIA machines, ensure:

* `nvidia-smi` works
* your torch build supports CUDA
