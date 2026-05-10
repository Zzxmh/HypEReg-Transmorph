# Scripts

Analysis and post-processing scripts for HypEReg-TransMorph.

## bootstrap_ci.py

Computes bootstrap 95% confidence intervals (B=10,000, seed=0) for five core IXI metrics across five models.

**Output:** `IXI/Results/comprehensive/bootstrap_ci_table_s6.csv` (Supplementary Table S6)

```bash
python scripts/bootstrap_ci.py
```

## oasis_downstream.py

Runs the OASIS zero-shot **multi-atlas label fusion** experiment (majority voting; six atlases, 20 test targets).

**Output:** `OASIS/Eval_Results/downstream/d1_per_target.csv`, `d1_summary.csv`

```bash
python scripts/oasis_downstream.py                         # all three models
python scripts/oasis_downstream.py --models "HypEReg-TransMorph (ZS)"  # one model
```

**Main-text Figure~6 (qualitative):** axial (or other axis) T1 with ground-truth vs fused segmentations requires the same `OASIS/data/` layout plus GPU checkpoints. Fused maps are cached under `OASIS/Eval_Results/downstream/fused_cache/` for fast re-rendering.

```bash
python figures/render_downstream_qualitative.py --target-ids 440 444 448 --device cuda
# or: python figures/visualize_downstream_segmentation.py   # default three targets, 3x4 grid
```

Table-style bar chart only (no volumes): `python figures/visualize_downstream_segmentation_bars.py`

## fill_downstream_results.py

Fills multi-atlas table rows in `draft/article.tex` from `d1_summary.csv`.

```bash
python scripts/fill_downstream_results.py --dry-run   # preview changes
python scripts/fill_downstream_results.py            # apply changes
```

## oasis_roi_analysis.py

Per-ROI Jacobian cleanliness analysis (hippocampus, lateral ventricles, thalamus) for supplementary Section S7.

**Requires:** displacement field `.npz` files under `OASIS/data/Submit/submission/*/task_03/`.  
Regenerate with: `python OASIS/export_displacements.py --model-id <id>`

```bash
python scripts/oasis_roi_analysis.py
```
