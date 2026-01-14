from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from heat_hpc.profiling.timers import StepTimer
from heat_hpc.solver.kernels import heat_step_5pt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-GPU 2D heat equation solver (explicit 5-point stencil).")
    p.add_argument("--N", type=int, default=2048, help="Grid size (NxN).")
    p.add_argument("--steps", type=int, default=1000, help="Number of time steps.")
    p.add_argument("--alpha", type=float, default=1.0, help="Diffusivity alpha.")
    p.add_argument("--dx", type=float, default=1.0, help="Grid spacing.")
    p.add_argument("--dt", type=str, default="auto", help='Time step, or "auto" for stable default.')
    p.add_argument("--device", type=str, default="cuda:0", help='Device, e.g. "cuda:0" or "cpu".')
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float32", help="Floating dtype.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--benchmark", action="store_true", help="Print timing summary.")
    p.add_argument("--warmup", type=int, default=20, help="Warmup steps (not timed).")
    p.add_argument("--log_every", type=int, default=100, help="Log every N steps.")
    p.add_argument("--check_cpu", type=int, default=0, help="If 1, run a small CPU reference check (for small N).")
    p.add_argument("--cpu_check_N_max", type=int, default=512, help="Max N allowed for CPU check.")

    # New: outputs
    p.add_argument("--out_json", type=str, default="", help="Write run summary JSON to this path.")
    p.add_argument(
        "--out_step_csv",
        type=str,
        default="",
        help="Write per-step timings CSV to this path (step,ms).",
    )

    # New: solution-profile-over-time
    p.add_argument(
        "--centerline_every",
        type=int,
        default=0,
        help="If >0, save the center row every K steps for visualization.",
    )
    p.add_argument(
        "--out_centerline_npz",
        type=str,
        default="",
        help="Where to write centerline NPZ (contains centerlines[T, N] and steps[T]).",
    )
    return p.parse_args()


def _pick_dt(alpha: float, dx: float) -> float:
    # Explicit 2D stability: c = alpha*dt/dx^2 <= 1/4. Choose 0.24 for margin.
    c = 0.24
    return c * (dx * dx) / alpha


def _dtype_from_str(s: str) -> torch.dtype:
    return torch.float32 if s == "float32" else torch.float64


def _ensure_parent(path: str) -> None:
    if not path:
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def run_single(
    N: int,
    steps: int,
    alpha: float,
    dx: float,
    dt: float,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    log_every: int,
    benchmark: bool,
    out_step_csv: str = "",
    centerline_every: int = 0,
    out_centerline_npz: str = "",
) -> dict:
    # Initial condition: a hot square in the middle
    u = torch.zeros((N, N), device=device, dtype=dtype)
    u_new = torch.empty_like(u)

    s0 = N // 2 - N // 10
    s1 = N // 2 + N // 10
    u[s0:s1, s0:s1] = 1.0

    c = alpha * dt / (dx * dx)
    if c > 0.25:
        raise ValueError(f"Unstable dt: c=alpha*dt/dx^2={c:.6f} > 0.25. Use smaller dt.")

    # Warmup (not timed)
    for _ in range(max(0, warmup)):
        heat_step_5pt(u, u_new, c)
        u, u_new = u_new, u

    # Optional centerline recording
    record_center = (centerline_every > 0 and out_centerline_npz != "")
    centerlines = []
    center_steps = []
    center_idx = N // 2

    # Timed run
    timer = StepTimer(device=device)
    times_ms = []

    if out_step_csv:
        _ensure_parent(out_step_csv)
        with open(out_step_csv, "w", encoding="utf-8") as f:
            f.write("step,ms\n")

    for step in range(steps):
        timer.start()
        heat_step_5pt(u, u_new, c)
        u, u_new = u_new, u
        ms = timer.stop_ms()
        times_ms.append(ms)

        if out_step_csv:
            with open(out_step_csv, "a", encoding="utf-8") as f:
                f.write(f"{step+1},{ms:.6f}\n")

        if record_center and ((step + 1) % centerline_every == 0 or (step + 1) == steps):
            row = u[center_idx, :].detach().to("cpu").float().numpy().copy()
            centerlines.append(row)
            center_steps.append(step + 1)



        if (step + 1) % log_every == 0 or step == 0 or (step + 1) == steps:
            maxv = float(u.max().item())
            minv = float(u.min().item())
            print(f"[step {step+1:6d}/{steps}] ms={ms:8.3f}  min={minv:+.4e}  max={maxv:+.4e}")

    # Save centerline NPZ
    if record_center:
        _ensure_parent(out_centerline_npz)
        arr = np.stack(centerlines, axis=0) if centerlines else np.zeros((0, N), dtype=np.float32)
        st = np.array(center_steps, dtype=np.int32)
        np.savez_compressed(out_centerline_npz, centerlines=arr, steps=st, N=np.int32(N), dt=np.float64(dt), c=np.float64(c))

    # Summary
    t = torch.tensor(times_ms)
    out = {
        "N": N,
        "steps": steps,
        "dt": dt,
        "c": float(c),
        "ms_mean": float(t.mean().item()),
        "ms_p50": float(t.median().item()),
        "ms_p90": float(t.kthvalue(int(math.ceil(0.90 * len(t)))).values.item()) if len(t) > 1 else float(t.item()),
        "ms_min": float(t.min().item()),
        "ms_max": float(t.max().item()),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
    }

    if benchmark:
        print("\nTiming summary (per step):")
        print(f"  mean: {out['ms_mean']:.3f} ms")
        print(f"  p50 : {out['ms_p50']:.3f} ms")
        print(f"  p90 : {out['ms_p90']:.3f} ms")
        print(f"  min : {out['ms_min']:.3f} ms")
        print(f"  max : {out['ms_max']:.3f} ms")
        print(f"  c   : {out['c']:.6f} (alpha*dt/dx^2)")

    return out


@torch.no_grad()
def cpu_reference_check(alpha: float, dx: float, dt: float, dtype: torch.dtype) -> None:
    N = 256
    steps = 50

    u_cpu = torch.zeros((N, N), device="cpu", dtype=dtype)
    u_cpu_new = torch.empty_like(u_cpu)
    s0 = N // 2 - N // 10
    s1 = N // 2 + N // 10
    u_cpu[s0:s1, s0:s1] = 1.0

    if not torch.cuda.is_available():
        print("CPU check skipped: CUDA not available.")
        return
    u_gpu = u_cpu.to("cuda")
    u_gpu_new = torch.empty_like(u_gpu)

    c = alpha * dt / (dx * dx)
    for _ in range(steps):
        heat_step_5pt(u_cpu, u_cpu_new, c)
        u_cpu, u_cpu_new = u_cpu_new, u_cpu
        heat_step_5pt(u_gpu, u_gpu_new, c)
        u_gpu, u_gpu_new = u_gpu_new, u_gpu

    diff = (u_cpu - u_gpu.cpu()).abs()
    print("\nCPU reference check:")
    print(f"  N={N}, steps={steps}, max_abs_diff={float(diff.max().item()):.6e}, mean_abs_diff={float(diff.mean().item()):.6e}")


def main() -> None:
    args = _parse_args()

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = _dtype_from_str(args.dtype)

    if args.dt == "auto":
        dt = _pick_dt(args.alpha, args.dx)
    else:
        dt = float(args.dt)

    if int(args.check_cpu) == 1:
        if args.N > args.cpu_check_N_max:
            print(f"CPU check not run: N={args.N} > cpu_check_N_max={args.cpu_check_N_max}.")
        else:
            cpu_reference_check(args.alpha, args.dx, dt, dtype)

    print("Single-GPU Heat Solver")
    print(f"  device: {args.device}")
    print(f"  dtype : {args.dtype}")
    print(f"  N     : {args.N}")
    print(f"  steps : {args.steps}")
    print(f"  alpha : {args.alpha}")
    print(f"  dx    : {args.dx}")
    print(f"  dt    : {dt} (c={args.alpha*dt/(args.dx*args.dx):.6f})")

    summary = run_single(
        N=args.N,
        steps=args.steps,
        alpha=args.alpha,
        dx=args.dx,
        dt=dt,
        device=args.device,
        dtype=dtype,
        warmup=args.warmup,
        log_every=args.log_every,
        benchmark=bool(args.benchmark),
        out_step_csv=args.out_step_csv,
        centerline_every=args.centerline_every,
        out_centerline_npz=args.out_centerline_npz,
    )

    if args.out_json:
        _ensure_parent(args.out_json)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary JSON: {args.out_json}")


if __name__ == "__main__":
    main()
