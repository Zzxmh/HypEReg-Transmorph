# Scripts

Analysis and post-processing scripts for HypEReg-TransMorph.

## bootstrap_ci.py

Computes bootstrap 95% confidence intervals (B=10,000, seed=0) for five core IXI metrics across five models.

**Output:** `IXI/Results/comprehensive/bootstrap_ci_table_s6.csv` (Supplementary Table S6)

```bash
python scripts/bootstrap_ci.py
```

## oasis_downstream.py

Runs downstream experiments on the OASIS zero-shot setting:

- **D-1** Multi-atlas label fusion (N=5 atlases, 20 test targets, majority voting)
- **D-2** ROI volumetric reliability (Jacobian integration + ICC(3,1))

**Output:** `OASIS/Eval_Results/downstream/d1_per_target.csv`, `d1_summary.csv`, `d2_per_target_roi.csv`, `d2_icc_summary.csv`

```bash
python scripts/oasis_downstream.py                         # all three models
python scripts/oasis_downstream.py --models "HypEReg-TransMorph (ZS)"  # one model
```

## fill_downstream_results.py

Fills `PLACEHOLDER` values in `draft/article.tex` from downstream experiment CSVs.

```bash
python scripts/fill_downstream_results.py --dry-run   # preview changes
python scripts/fill_downstream_results.py             # apply changes
```

## oasis_roi_analysis.py

Per-ROI Jacobian cleanliness analysis (hippocampus, lateral ventricles, thalamus) for Supplementary Table S7.

**Requires:** displacement field `.npz` files under `OASIS/data/Submit/submission/*/task_03/`.  
Regenerate with: `python OASIS/export_displacements.py --model-id <id>`

```bash
python scripts/oasis_roi_analysis.py
```
