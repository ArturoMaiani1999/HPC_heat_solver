from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description="Plot heat solver centerline profile vs time.")
    p.add_argument("npz", type=str, help="Path to *_centerline.npz produced by heat_single.py")
    p.add_argument("--out", type=str, default="", help="Optional output PNG path")
    p.add_argument("--title", type=str, default="", help="Optional plot title")
    args = p.parse_args()

    data = np.load(args.npz)
    centerlines = data["centerlines"]  # shape [T, N]
    steps = data["steps"]              # shape [T]
    N = int(data["N"])
    dt = float(data["dt"])
    c = float(data["c"])

    if centerlines.size == 0:
        raise SystemExit("No centerlines found. Did you run with --centerline_every > 0?")

    # Plot: time (rows) vs x (cols)
    plt.figure()
    plt.imshow(centerlines, aspect="auto", origin="lower")
    plt.colorbar(label="u(x, y=N/2)")
    plt.xlabel("x index")
    plt.ylabel("snapshot index (time)")
    title = args.title or f"Centerline evolution (N={N}, dt={dt:.3g}, c={c:.3g})"
    plt.title(title)

    # Annotate approximate physical time at last snapshot
    t_end = steps[-1] * dt
    plt.figtext(0.5, 0.01, f"last step={int(steps[-1])}, t_end≈{t_end:.3g}", ha="center")

    if args.out:
        Path(args.out).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Wrote: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
