from __future__ import annotations

import torch


def heat_step_5pt(
    u: torch.Tensor,
    u_new: torch.Tensor,
    c: float,
) -> None:
    """One explicit heat-equation step using a 5-point stencil.

    Updates interior points, leaves boundary points unchanged.

    Args:
        u:     (N, N) current field
        u_new: (N, N) output buffer (will be overwritten)
        c:     alpha * dt / dx^2
    """
    # Copy boundary as-is (Dirichlet fixed boundaries)
    u_new.copy_(u)

    # Interior update
    # u_new[i,j] = u[i,j] + c*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]-4*u[i,j])
    center = u[1:-1, 1:-1]
    up = u[0:-2, 1:-1]
    down = u[2:, 1:-1]
    left = u[1:-1, 0:-2]
    right = u[1:-1, 2:]

    u_new[1:-1, 1:-1] = center + c * (up + down + left + right - 4.0 * center)
