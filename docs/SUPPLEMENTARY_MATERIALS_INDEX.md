# Supplementary Materials Index

This file defines the supplementary package for the current HER-TransMorph manuscript revision and maps each supplement item to source files in this repository.

## 1) Supplementary items cited in manuscript

Manuscript source: `draft/template_ixi_only.tex`

- **Table S1**: Extended descriptive metric panel.
- **Table S2**: Paired inferential statistics (Wilcoxon + BH-FDR, with rank-biserial signed effect).
- **Table S3**: Grouped-VOI label mapping used for grouped Dice.
- **Figure S1**: IXI quantitative overview figure.

## 2) Supplementary tables: source artifacts

### Table S1 (descriptive panel)
- Built from manuscript-integrated summary values and IXI exports.
- Primary data sources:
  - `IXI/Results/comprehensive/model_summary.csv`
  - `IXI/Results/comprehensive/sig_matrix.csv`

### Table S2 (inferential statistics)
- Primary exported data:
  - `IXI/Results/comprehensive/sig_matrix_extended.csv`
  - `IXI/Results/comprehensive/sig_matrix_uploaded_ckpt.csv`
- Generation scripts:
  - `IXI/analysis_comprehensive/export_inferential_stats.py`
  - `IXI/analysis_comprehensive/export_uploaded_weight_stats.py`

### Table S3 (grouped VOI mapping)
- Mapping source logic:
  - `IXI/TransMorph/utils.py` (`process_label`)
  - `IXI/analysis.py`
  - `IXI/analysis_trans.py`

## 3) Supplementary figures: source artifacts

### Figure S1
- Figure file:
  - `figures/fig5_metrics.pdf`
- Generation script:
  - `figures/regenerate_figures.py`

### Additional qualitative/Jacobian visuals
- Figure files:
  - `figures/fig2_qualitative.pdf`
  - `figures/fig3_gridwarp.pdf`
  - `figures/fig4_jacobian.pdf`
- Generation script:
  - `figures/regenerate_figures.py`

## 4) Supplementary code package (reproduction-critical)

- Evaluation and metrics:
  - `IXI/eval_any.py`
  - `IXI/eval_configs.yaml`
  - `IXI/metrics_full.py`
  - `OASIS/eval_oasis.py`
  - `OASIS/eval_configs.yaml`
- Figure/statistics:
  - `figures/regenerate_figures.py`
  - `IXI/analysis_comprehensive/`
- Runtime profiling:
  - `IXI/profile_forward_runtime_memory.py`

## 5) Supplementary data package (recommended for archival upload)

Recommended archive content (DOI repository):
- `IXI/Results/comprehensive/` (final CSV summaries and inferential tables)
- `IXI/Results/uploaded_weights_light/` (light per-case paired analysis inputs)
- `IXI/Eval_Results/*/per_case.csv` (if size/privacy/license permits)
- A manifest file with:
  - repository commit hash,
  - generation date,
  - script command lines,
  - checksums for large files.

## 6) Recommended supplementary package layout

Example zip layout for submission/release:

```text
supplementary/
  README_SUPPLEMENTARY.md
  tables/
    Table_S1_descriptive.csv
    Table_S2_inferential.csv
    Table_S3_grouped_voi_mapping.csv
  figures/
    Figure_S1_metrics_overview.pdf
    Figure_S2_qualitative_cases.pdf
    Figure_S3_gridwarp.pdf
    Figure_S4_jacobian.pdf
  code/
    export_inferential_stats.py
    export_uploaded_weight_stats.py
    regenerate_figures.py
    profile_forward_runtime_memory.py
  manifests/
    artifact_manifest.csv
    checksums_sha256.txt
```
