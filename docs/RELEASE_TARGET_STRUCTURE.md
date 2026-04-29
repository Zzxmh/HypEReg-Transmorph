# Release Target Structure and Migration Map

This document proposes a clearer public-facing repository layout while preserving current code paths.

## 1) Target structure (release branch)

```text
.
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ requirements-stable.txt
├─ docs/
│  ├─ MDPI_JIMAGING_COMPLIANCE_CHECKLIST.md
│  ├─ SUPPLEMENTARY_MATERIALS_INDEX.md
│  ├─ RELEASE_CLEANUP_AUDIT.md
│  ├─ RELEASE_TARGET_STRUCTURE.md
│  └─ PREPRINT_SUBMISSION_STRATEGY.md
├─ TransMorph/
├─ TransMorph_affine/
├─ TransMorph_PSC/
├─ Baseline_registration_models/
├─ Baseline_Transformers/
├─ IXI/
├─ OASIS/
├─ RaFD/
├─ figures/
├─ Docker/
├─ example_imgs/
└─ paper/               # optional: source-only paper folder
```

## 2) What to move or trim

### A) Paper assets

- Current location: `draft/`
- Recommended public layout:
  - Keep source files only:
    - `draft/template_ixi_only.tex`
    - `draft/refs.bib`
    - `draft/build_drafts.py`
  - Move into optional `paper/` folder or keep under `draft/` but delete build byproducts.

### B) Internal workflow documents

- Archive/private (not default public branch):
  - `REVISION_HANDOFF.md`
  - `SANITY_PASS_STAGEA.md`
  - `registration_notice.md`
  - `draft/submission_package/` (unless explicitly publishing full submission package)

### C) Artifacts and checkpoints

- Do not track in git:
  - `*.pth`, `*.pth.tar`, `*.ckpt`
  - local eval logs/intermediate outputs
- Publish externally with version pinning:
  - GitHub Release assets or Zenodo/Figshare DOI package
  - include `sha256` checksums and asset manifest

## 3) Migration map (old -> target)

| Current path | Target path | Action |
|---|---|---|
| `draft/` | `paper/` or trimmed `draft/` | Keep source-only files, remove build outputs |
| `IXI/Results/*` | `artifacts (external)` | Export curated CSV set to DOI archive |
| `IXI/Eval_Results/*` | `artifacts (external)` | Optional curated release, path-sanitized metadata |
| `training_*.log` | removed | Generated logs not part of release |
| `.venv/` | removed | Local environment only |

## 4) Minimal non-breaking strategy

To avoid breaking existing scripts, keep current code directories and apply only:
1. cleanup + ignore rules,
2. documentation improvements,
3. external artifact hosting,
4. optional `paper/` extraction without renaming core code folders.

This gives a clean release with minimal refactor risk.
