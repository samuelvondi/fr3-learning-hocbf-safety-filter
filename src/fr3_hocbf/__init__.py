"""FR3 learned-HOCBF-gain pipeline. Run scripts from src/ as `python -m fr3_hocbf.<stage>.<script>`."""

from pathlib import Path

# Repository root (src/fr3_hocbf/__init__.py -> parents[2]); used for include/urdf and config paths.
REPO_ROOT = Path(__file__).resolve().parents[2]
