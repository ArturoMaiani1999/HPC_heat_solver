"""Stencil kernels (core).

For now, this module re-exports the current implementation from heat_hpc.solver.kernels.
Later we can move/expand kernels here without changing call-sites.
"""

from heat_hpc.solver.kernels import heat_step_5pt  # noqa: F401
