from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

DTypeStr = Literal["float32", "float64"]


@dataclass(frozen=True)
class HeatConfig:
    # Grid
    N: int = 2048
    alpha: float = 1.0
    dx: float = 1.0
    dt: float | Literal["auto"] = "auto"

    # Run
    steps: int = 1000
    seed: int = 0
    device: str = "cuda:0"
    dtype: DTypeStr = "float32"

    # Benchmarking
    benchmark: bool = False
    warmup: int = 20

    # Correctness
    check_cpu: bool = False
    cpu_check_N_max: int = 512  # safety cap to avoid accidental huge CPU checks

    # Output
    log_every: int = 100
