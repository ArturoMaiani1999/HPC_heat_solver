
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-heat-hpc}"
PKG_DIR="$ROOT_DIR/src/heat_hpc"

echo "Creating repo skeleton at: $ROOT_DIR"

# Directories
mkdir -p "$ROOT_DIR"/{src,tests,scripts,docs,experiments/{configs,notebooks,results}}
mkdir -p "$PKG_DIR"/{solver,dist,profiling,io,utils}

# Top-level files
touch "$ROOT_DIR"/{README.md,LICENSE,.gitignore,.editorconfig,pyproject.toml}

# Package files
touch "$PKG_DIR"/__init__.py
touch "$PKG_DIR"/{config.py,grid.py}
touch "$PKG_DIR"/solver/{__init__.py,heat_single.py,heat_dist.py,kernels.py}
touch "$PKG_DIR"/dist/{__init__.py,init.py,halo.py}
touch "$PKG_DIR"/profiling/{__init__.py,timers.py,traces.py}
touch "$PKG_DIR"/io/{__init__.py,checkpoints.py,viz.py}
touch "$PKG_DIR"/utils/{__init__.py,logging.py,reproducibility.py}

# Scripts (empty placeholders for now)
touch "$ROOT_DIR"/scripts/{run_single.sh,run_dist_rank0.sh,run_dist_rank1.sh,bench_single.sh,bench_dist.sh}

# Tests
touch "$ROOT_DIR"/tests/{test_correctness_small.py,test_halo_exchange.py}

# Docs
touch "$ROOT_DIR"/docs/{roadmap.md,theory.md,troubleshooting.md}

# Make bash scripts executable
chmod +x "$ROOT_DIR"/scripts/*.sh

# Put a minimal marker in results folder so it exists (optional)
touch "$ROOT_DIR"/experiments/results/.gitkeep

echo "Done."
echo
echo "Next steps:"
echo "  1) cd $ROOT_DIR"
echo "  2) Open README.md and pyproject.toml"
echo "  3) We'll fill scripts/ and src/heat_hpc/* next"
