#!/usr/bin/env bash
set -euo pipefail

# Refactor the repo into a didactic structure while keeping backwards compatibility.
# Run from the repository root (the folder that contains: docs/, experiments/, scripts/, src/, tests/).

ROOT_DIR="$(pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

# Sanity checks
[[ -d "scripts" ]] || die "Expected ./scripts folder. Run this from the repo root."
[[ -d "src/heat_hpc" ]] || die "Expected ./src/heat_hpc package folder. Run this from the repo root."

# Helpers
move_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    if [[ -e "$dst" ]]; then
      info "Skip move (dest exists): $src -> $dst"
    else
      mkdir -p "$(dirname "$dst")"
      mv "$src" "$dst"
      info "Moved: $src -> $dst"
    fi
  else
    info "Skip move (missing): $src"
  fi
}

write_file_if_missing() {
  local path="$1"
  local content="$2"
  if [[ -e "$path" ]]; then
    info "Exists: $path"
  else
    mkdir -p "$(dirname "$path")"
    printf "%s" "$content" > "$path"
    info "Created: $path"
  fi
}

make_executable_if_exists() {
  local path="$1"
  if [[ -e "$path" ]]; then
    chmod +x "$path"
  fi
}

# ------------------------------------------------------------------------------
# 1) Create didactic top-level folders
# ------------------------------------------------------------------------------
mkdir -p basics lessons
info "Ensured basics/ and lessons/ exist."

write_file_if_missing "basics/README.md" \
"# Basics (teaching scripts)

This folder contains minimal, single-file implementations used for lessons and live coding.
It intentionally duplicates some logic from the library for didactic clarity.

Planned files:
- 00_numpy_heat.py
- 01_torch_cpu_heat.py
- 02_torch_gpu_heat.py
- 03_stencil_kernels.py
"

write_file_if_missing "lessons/README.md" \
"# Lessons

Lesson plan / lecture notes (markdown) that reference code in:
- basics/ (minimal teaching scripts)
- src/heat_hpc/ (the real library)
- scripts/ (repro commands)

Planned lessons:
- 01_cpu_basics.md
- 02_gpu_basics.md
- 03_profiling.md
- 04_io_and_viz.md
- 05_distributed_intro.md
- 06_mpi_task_parallel.md
"

# Empty lesson placeholders
for f in \
  "lessons/01_cpu_basics.md" \
  "lessons/02_gpu_basics.md" \
  "lessons/03_profiling.md" \
  "lessons/04_io_and_viz.md" \
  "lessons/05_distributed_intro.md" \
  "lessons/06_mpi_task_parallel.md"
do
  write_file_if_missing "$f" "# $(basename "$f" .md)\n\n(TODO)\n"
done

# Empty basics placeholders
for f in \
  "basics/00_numpy_heat.py" \
  "basics/01_torch_cpu_heat.py" \
  "basics/02_torch_gpu_heat.py" \
  "basics/03_stencil_kernels.py"
do
  write_file_if_missing "$f" "# TODO: lesson script\n"
done

# ------------------------------------------------------------------------------
# 2) Refactor scripts/ into subfolders and move existing scripts
# ------------------------------------------------------------------------------
mkdir -p scripts/single scripts/dist scripts/viz
info "Ensured scripts/single, scripts/dist, scripts/viz exist."

# Move existing scripts into categorized folders
move_if_exists "scripts/run_single.sh"      "scripts/single/run_single.sh"
move_if_exists "scripts/bench_single.sh"    "scripts/single/bench_single.sh"
move_if_exists "scripts/plot_centerline.py" "scripts/viz/plot_centerline.py"

move_if_exists "scripts/bench_dist.sh"      "scripts/dist/bench_dist.sh"
move_if_exists "scripts/run_dist_rank0.sh"  "scripts/dist/run_rank0.sh"
move_if_exists "scripts/run_dist_rank1.sh"  "scripts/dist/run_rank1.sh"

# Create backward-compatible wrappers at old locations (so old READMEs keep working)
write_file_if_missing "scripts/run_single.sh" \
"#!/usr/bin/env bash
set -euo pipefail
DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\"
exec \"\$DIR/single/run_single.sh\" \"\$@\"
"
write_file_if_missing "scripts/bench_single.sh" \
"#!/usr/bin/env bash
set -euo pipefail
DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\"
exec \"\$DIR/single/bench_single.sh\" \"\$@\"
"
write_file_if_missing "scripts/plot_centerline.py" \
"#!/usr/bin/env python
# Wrapper kept for backwards compatibility.
import runpy
runpy.run_path('scripts/viz/plot_centerline.py', run_name='__main__')
"

write_file_if_missing "scripts/bench_dist.sh" \
"#!/usr/bin/env bash
set -euo pipefail
DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\"
exec \"\$DIR/dist/bench_dist.sh\" \"\$@\"
"
write_file_if_missing "scripts/run_dist_rank0.sh" \
"#!/usr/bin/env bash
set -euo pipefail
DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\"
exec \"\$DIR/dist/run_rank0.sh\" \"\$@\"
"
write_file_if_missing "scripts/run_dist_rank1.sh" \
"#!/usr/bin/env bash
set -euo pipefail
DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)\"
exec \"\$DIR/dist/run_rank1.sh\" \"\$@\"
"

# Ensure executables
make_executable_if_exists "scripts/run_single.sh"
make_executable_if_exists "scripts/bench_single.sh"
make_executable_if_exists "scripts/bench_dist.sh"
make_executable_if_exists "scripts/run_dist_rank0.sh"
make_executable_if_exists "scripts/run_dist_rank1.sh"
make_executable_if_exists "scripts/single/run_single.sh"
make_executable_if_exists "scripts/single/bench_single.sh"
make_executable_if_exists "scripts/dist/bench_dist.sh"
make_executable_if_exists "scripts/dist/run_rank0.sh"
make_executable_if_exists "scripts/dist/run_rank1.sh"

# New empty helper scripts we’ll populate later
write_file_if_missing "scripts/dist/run_mpi_sweep.sh" \
"#!/usr/bin/env bash
set -euo pipefail
# TODO: mpirun wrapper to run task-parallel MPI sweep across nodes.
# Example (future):
# mpirun -np 2 --hostfile ~/mpi_hosts --map-by slot \\
#   python -m heat_hpc.dist.mpi_sweep --sizes 1024,2048 --steps 1000 --device cuda:0
echo \"TODO: implement MPI sweep launcher\"
exit 1
"
make_executable_if_exists "scripts/dist/run_mpi_sweep.sh"

write_file_if_missing "scripts/single/viz_single.sh" \
"#!/usr/bin/env bash
set -euo pipefail
# TODO: convenience wrapper for visualization runs (warmup=0, frequent snapshots).
echo \"TODO: implement viz run wrapper\"
exit 1
"
make_executable_if_exists "scripts/single/viz_single.sh"

# ------------------------------------------------------------------------------
# 3) Create new “advanced scenario” python modules (empty placeholders)
#    without breaking current code paths.
# ------------------------------------------------------------------------------
mkdir -p src/heat_hpc/core src/heat_hpc/single src/heat_hpc/dist
info "Ensured src/heat_hpc/{core,single,dist} exist."

write_file_if_missing "src/heat_hpc/core/__init__.py" \
"\"\"\"Core numerical building blocks (stencil, grids, ICs).

We keep these small and reusable for teaching and for the higher-level solvers.
\"\"\"
"

# Bridge: core/stencil.py re-exports current kernel(s) so we don't break anything yet.
write_file_if_missing "src/heat_hpc/core/stencil.py" \
"\"\"\"Stencil kernels (core).

For now, this module re-exports the current implementation from heat_hpc.solver.kernels.
Later we can move/expand kernels here without changing call-sites.
\"\"\"

from heat_hpc.solver.kernels import heat_step_5pt  # noqa: F401
"

write_file_if_missing "src/heat_hpc/core/init_conditions.py" \
"\"\"\"Initial conditions utilities (TODO).\"\"\"\n"

write_file_if_missing "src/heat_hpc/single/__init__.py" \
"\"\"\"Single-device entrypoints.\"\"\"\n"

# Bridge entrypoint: single/run.py calls existing solver.heat_single main
write_file_if_missing "src/heat_hpc/single/run.py" \
"\"\"\"Single-device runner (didactic entrypoint).

This is a thin wrapper around the existing implementation in heat_hpc.solver.heat_single.
\"\"\"

from heat_hpc.solver.heat_single import main

if __name__ == \"__main__\":
    main()
"

write_file_if_missing "src/heat_hpc/dist/__init__.py" \
"\"\"\"Distributed scenarios.

- mpi_sweep: task-parallel (good over Wi-Fi)
- domain_decomp: halo exchange / domain decomposition (network-sensitive)
\"\"\"\n"

write_file_if_missing "src/heat_hpc/dist/mpi_sweep.py" \
"\"\"\"MPI task-parallel sweep (TODO).

Goal: distribute independent runs across ranks/nodes and aggregate results.
\"\"\"\n"

write_file_if_missing "src/heat_hpc/dist/domain_decomp.py" \
"\"\"\"Domain decomposition + halo exchange (TODO).

Goal: split the grid across ranks and exchange halos every timestep.
\"\"\"\n"

# ------------------------------------------------------------------------------
# 4) Optional cleanup hint: __pycache__ and egg-info should be gitignored
# ------------------------------------------------------------------------------
info "Done."

cat <<'EOF'

Next actions (recommended):
  1) Review changes:
       git status
       git diff

  2) Verify old commands still work (wrappers):
       bash scripts/run_single.sh 512 200 float32
       bash scripts/bench_single.sh

  3) New locations:
       scripts/single/run_single.sh
       scripts/single/bench_single.sh
       scripts/viz/plot_centerline.py
       scripts/dist/...

EOF
