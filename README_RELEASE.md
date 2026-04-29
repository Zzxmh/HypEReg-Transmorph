# HypEReg-TransMorph / TransMorph Release README

This document is the public-release oriented guide for this repository (code, supplementary artifacts, and reproducibility workflow).

For repository overview and maintainer-facing entry point, see `README.md`.

## 1) Scope

This repository contains:
- **IXI workflow** (`IXI/`): HypEReg/TransMorph code under `IXI/TransMorph/`, unified evaluation and adapters under `IXI/adapters/`, CNN/transformer baselines under `IXI/Baseline_registration_methods/` and `IXI/Baseline_Transformers/`, optional traditional baselines under `IXI/Baseline_traditional_methods/`
- **OASIS workflow** (`OASIS/`) for deferred / optional experiments
- **Manuscript** sources under `draft/`
- **Documentation** under `docs/` (supplementary index, compliance checklist, cleanup notes)
- **Figure pipelines** under `figures/` and `IXI/analysis_comprehensive/`

Current manuscript-oriented benchmark scope is **IXI atlas-to-subject protocol**.

## 2) Environment Setup

### Option A: general environment

```bash
pip install -r requirements.txt
```

### Option B: stable pinned environment (recommended for reproducibility)

```bash
pip install -r requirements-stable.txt
```

### Option C: full IXI evaluation stack

```bash
pip install -r IXI/requirements-ixi-eval.txt
```

## 3) Data and Weights

- **IXI / OASIS data layout:** use a local root (e.g. `IXI_data/`, `OASIS/data/`) and `PreprocessingMRI.md`; optional inventory: `docs/IXI_OASIS_FILE_CLASSIFICATION.md`
- MRI preprocessing notes: `PreprocessingMRI.md`

Release policy:
- large checkpoints (`*.pth`, `*.pth.tar`) and large datasets should be published as release assets or DOI archives, not tracked in git.
- provide a checksum manifest (`sha256`) for all downloadable binaries.

## 4) Quickstart Reproduction

### 4.1 Run IXI evaluation for selected models

```bash
python IXI/run_full_eval.py --config IXI/eval_configs.yaml --only transmorph_her,transmorph_original
```

### 4.2 Export inferential tables

```bash
python -m IXI.analysis_comprehensive.export_inferential_stats --eval_root IXI/Eval_Results --out_csv IXI/Results/comprehensive/sig_matrix_extended.csv
python IXI/analysis_comprehensive/export_uploaded_weight_stats.py --eval_root IXI/Eval_Results --light_root IXI/Results/uploaded_weights_light --out_csv IXI/Results/comprehensive/sig_matrix_uploaded_ckpt.csv
```

### 4.3 Regenerate manuscript figures

```bash
python figures/regenerate_figures.py --fig_dir figures --subject subject_1.pkl --models transmorph_her,transmorph_original,midir
```

### 4.4 Profile pure forward runtime/memory

```bash
python IXI/profile_forward_runtime_memory.py --subject subject_1.pkl --repeats 20
```

## 5) Reproducibility Matrix (script -> output)

| Script | Main outputs |
|---|---|
| `IXI/run_full_eval.py` | `IXI/Eval_Results/<model>/per_case.csv`, `meta.json` |
| `IXI/analysis_comprehensive/export_inferential_stats.py` | `IXI/Results/comprehensive/sig_matrix_extended.csv` |
| `IXI/analysis_comprehensive/export_uploaded_weight_stats.py` | `IXI/Results/comprehensive/sig_matrix_uploaded_ckpt.csv` |
| `figures/regenerate_figures.py` | `figures/fig2_qualitative.pdf`, `figures/fig3_gridwarp.pdf`, `figures/fig4_jacobian.pdf`, `figures/fig5_metrics.pdf` |
| `IXI/profile_forward_runtime_memory.py` | console profile stats for pure forward cost |

## 6) Supplementary Materials and Journal Compliance

- Supplementary index: `docs/SUPPLEMENTARY_MATERIALS_INDEX.md`
- MDPI/J. Imaging checklist: `docs/MDPI_JIMAGING_COMPLIANCE_CHECKLIST.md`
- Cleanup policy before release: `docs/RELEASE_CLEANUP_AUDIT.md`
- Target release structure: `docs/RELEASE_TARGET_STRUCTURE.md`
- Upstream attribution and redistribution note: `NOTICE`

## 7) Repository Hygiene Before Tagging a Public Release

1. Remove generated logs/build outputs.
2. Ensure `.gitignore` excludes local env, checkpoints, and temporary artifacts.
3. Sanitize absolute local paths in generated metadata.
4. Validate quickstart commands from a clean environment.
5. Attach external artifact links + checksums to release notes.

## 8) Docker

See `Docker/README.md` for containerized workflows.

## 9) Citation

If this code is useful for your work, please cite both the upstream TransMorph paper and your HypEReg manuscript/preprint:

```bibtex
@article{chen2022transmorph,
  title = {TransMorph: Transformer for unsupervised medical image registration},
  journal = {Medical Image Analysis},
  pages = {102615},
  year = {2022},
  doi = {10.1016/j.media.2022.102615},
  author = {Junyu Chen and Eric C. Frey and Yufan He and William P. Segars and Ye Li and Yong Du}
}
```

At release time, keep attribution to the upstream repository in `README.md` and retain `LICENSE` + `NOTICE` in redistributed copies.
