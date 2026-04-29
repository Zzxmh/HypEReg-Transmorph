# Release Cleanup Audit

This document classifies repository contents for open-source release preparation.

## Classification labels

- **KEEP**: required for users to run code or reproduce paper outputs.
- **ARCHIVE**: keep in private/internal storage, not in public default branch.
- **DELETE_FROM_RELEASE**: generated clutter not needed for release.
- **EXTERNALIZE**: large files best hosted outside git (DOI/release assets/cloud).

## 1) Root-level actions

| Path | Action | Notes |
|---|---|---|
| `.venv/` | DELETE_FROM_RELEASE | Local env only. |
| `training_psc_run.log`, `training_psc_gpu.log` | DELETE_FROM_RELEASE | Generated run logs. |
| `MDPI_template_ACS.zip` | ARCHIVE | Authoring artifact, not runtime dependency. |
| `REVISION_HANDOFF.md`, `SANITY_PASS_STAGEA.md` | ARCHIVE | Internal process documents. |
| `registration_notice.md` | ARCHIVE | Internal submission workflow note. |
| `README.md`, `LICENSE`, `requirements*.txt` | KEEP | Public entry and environment files. |

## 2) Draft/paper folder actions

| Path pattern | Action | Notes |
|---|---|---|
| `draft/*.aux`, `draft/*.log`, `draft/*.out`, `draft/*.blg`, `draft/*.bbl` | DELETE_FROM_RELEASE | Compile byproducts. |
| `draft/submission_draft_backup_*/` | DELETE_FROM_RELEASE | Local backups and duplicated artifacts. |
| `draft/submission_package/` | ARCHIVE | Keep only if you publish submission bundle. |
| `draft/template_ixi_only.tex`, `draft/refs.bib`, `draft/build_drafts.py` | KEEP | Source-level paper files. |

## 3) Model/data artifact actions

| Path pattern | Action | Notes |
|---|---|---|
| `**/*.pth`, `**/*.pth.tar`, `**/*.ckpt` | EXTERNALIZE | Publish via release assets/Zenodo with checksums. |
| `IXI_data/*.pkl` | EXTERNALIZE | Dataset-derived assets should not live in git. |
| `IXI/Eval_Results/` | EXTERNALIZE | Large outputs; publish curated subset if needed. |
| `IXI/Results/comprehensive/interim/` | DELETE_FROM_RELEASE | Intermediate outputs only. |
| `IXI/Results/comprehensive_bayes_eval/` | ARCHIVE | Keep only final curated outputs public. |
| `IXI/Results/uploaded_weights_light/` | EXTERNALIZE | Good supplementary artifact; avoid git bloat. |

## 4) Metadata sanitization

Before release, sanitize generated metadata files containing local absolute paths:

- `IXI/Eval_Results/*/meta.json`
- Any `summary.json` with local `F:\\...` path strings

Recommended: regenerate metadata with relative paths, or strip machine-specific fields.

## 5) Keep list (public release minimum)

- Core code: `TransMorph/`, `TransMorph_affine/`, `TransMorph_PSC/`, `Baseline_registration_models/`, `Baseline_Transformers/`, `IXI/`, `OASIS/`, `RaFD/`, `figures/`, `Docker/`
- Docs: `README.md`, `docs/`, dataset guides (`IXI/TransMorph_on_IXI.md`, `OASIS/TransMorph_on_OASIS.md`, `PreprocessingMRI.md`)
- Configs/scripts for reproduction:
  - `IXI/eval_configs.yaml`, `IXI/eval_any.py`, `IXI/analysis_comprehensive/`
  - `OASIS/eval_configs.yaml`, `OASIS/eval_oasis.py`
  - `figures/regenerate_figures.py`

## 6) Immediate cleanup command checklist (manual)

Use a release branch and remove generated clutter before tagging:

1. Remove local env and caches.
2. Remove logs and temporary files.
3. Remove LaTeX compile byproducts/backups.
4. Ensure checkpoints are not tracked.
5. Verify `.gitignore` rules enforce the above.
6. Re-run a clean smoke test from documented commands.
