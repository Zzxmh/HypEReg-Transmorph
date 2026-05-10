"""Regenerate fig5_metrics.pdf using existing CSV data (no inference needed)."""
import sys, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "IXI"))

from pathlib import Path
from figures.regenerate_figures import render_fig5

out_path = Path(FIGURES_DIR) / "fig5_metrics.pdf"
render_fig5(out_path)
print(f"Regenerated: {out_path}")
