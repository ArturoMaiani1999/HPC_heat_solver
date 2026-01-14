from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class StepTimer:
    """CUDA-safe step timer.

    Use:
        t = StepTimer(device="cuda")
        t.start()
        ... do work ...
        ms = t.stop_ms()
    """

    device: str = "cuda"
    _t0: Optional[float] = None

    def start(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def stop_ms(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Timer was not started.")
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - self._t0) * 1000.0
