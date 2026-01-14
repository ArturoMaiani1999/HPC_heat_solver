from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mpi4py import MPI


@dataclass(frozen=True)
class Task:
    N: int
    steps: int
    dtype: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_one(task: Task, out_dir: Path, device: str, warmup: int, centerline_every: int) -> Dict:
    """
    Run one simulation on the local machine (and local GPU) by calling the existing single solver.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    host = socket.gethostname()

    tag = f"mpi_{host}_rank{rank}_{task.dtype}_N{task.N}_S{task.steps}"
    summary_json = out_dir / f"{tag}_summary.json"
    steps_csv = out_dir / f"{tag}_steps.csv"
    center_npz = out_dir / f"{tag}_centerline.npz"

    cmd = [
        "python", "-m", "heat_hpc.solver.heat_single",
        "--N", str(task.N),
        "--steps", str(task.steps),
        "--alpha", "1.0",
        "--dx", "1.0",
        "--dt", "auto",
        "--dtype", task.dtype,
        "--device", device,
        "--seed", "0",
        "--benchmark",
        "--warmup", str(warmup),
        "--log_every", str(task.steps),
        "--out_json", str(summary_json),
        "--out_step_csv", str(steps_csv),
        "--centerline_every", str(centerline_every),
        "--out_centerline_npz", str(center_npz),
    ]

    env = os.environ.copy()
    # Each node has one GPU; ensure the local process uses GPU 0 on that node.
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    subprocess.check_call(cmd, env=env)

    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    summary["host"] = host
    summary["rank"] = rank
    summary["_summary_json"] = str(summary_json)
    summary["_steps_csv"] = str(steps_csv)
    summary["_centerline_npz"] = str(center_npz)
    return summary


def rank0_schedule(comm: MPI.Comm, tasks: List[Task]) -> List[Dict]:
    """
    Simple dynamic scheduling:
    - rank 0 distributes tasks to workers
    - workers run tasks and send back a summary dict
    - rank 0 also works on tasks locally
    """
    size = comm.Get_size()
    results: List[Dict] = []

    next_task = 0

    # Prime workers
    for r in range(1, size):
        if next_task < len(tasks):
            comm.send(tasks[next_task], dest=r, tag=1)
            next_task += 1
        else:
            comm.send(None, dest=r, tag=0)

    # Rank 0 does remaining tasks locally (handled in main loop for simplicity)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="MPI task-parallel sweep for heat-hpc (useful even on Wi-Fi).")
    ap.add_argument("--out_dir", type=str, default="experiments/results", help="Where to write artifacts (local per node).")
    ap.add_argument("--device", type=str, default="cuda:0", help='Device string per node, typically "cuda:0" or "cpu".')
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--sizes", type=str, default="1024,2048,4096", help="Comma-separated N values.")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--centerline_every", type=int, default=50)
    ap.add_argument("--csv_name", type=str, default="mpi_sweep.csv")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    tasks = [Task(N=n, steps=args.steps, dtype=args.dtype) for n in sizes]

    # Dynamic scheduling:
    results: List[Dict] = []

    if rank == 0:
        next_task = 0

        # Prime workers
        for r in range(1, size):
            if next_task < len(tasks):
                comm.send(tasks[next_task], dest=r, tag=1)
                next_task += 1
            else:
                comm.send(None, dest=r, tag=0)

        # Rank 0 also works through remaining tasks
        while next_task < len(tasks):
            t = tasks[next_task]
            results.append(run_one(t, out_dir, args.device, args.warmup, args.centerline_every))
            next_task += 1

        # Collect worker results; when a worker returns, give it a new task until none left
        finished_workers = 0
        while finished_workers < (size - 1):
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=MPI.Status())
            src = msg.get("rank", None)

            results.append(msg)

            # give more work or stop the worker
            if next_task < len(tasks):
                comm.send(tasks[next_task], dest=src, tag=1)
                next_task += 1
            else:
                comm.send(None, dest=src, tag=0)
                finished_workers += 1

        # Write aggregate CSV (on rank 0)
        csv_path = out_dir / args.csv_name
        fieldnames = [
            "host", "rank", "device", "dtype", "N", "steps", "dt", "c",
            "ms_mean", "ms_p50", "ms_p90", "ms_min", "ms_max",
            "_summary_json", "_steps_csv", "_centerline_npz",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in sorted(results, key=lambda x: (x["N"], x["host"], x["rank"])):
                w.writerow({k: r.get(k, "") for k in fieldnames})

        print(f"[rank0] wrote: {csv_path}")

    else:
        # Worker loop
        while True:
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=MPI.Status())
            if task is None:
                break
            summary = run_one(task, out_dir, args.device, args.warmup, args.centerline_every)
            comm.send(summary, dest=0, tag=2)


if __name__ == "__main__":
    main()
