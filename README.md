# HypEReg-TransMorph &mdash; IXI evaluation and manuscript assets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Maintainers:** Mohan Xu, Shiyi Xu.

This repository is the **project codebase** for *HypEReg-TransMorph*: hyperelastic-regularized, folding-suppressed deformable registration built on the TransMorph backbone, with a reproducible **IXI atlas-to-subject** evaluation stack, inferential statistics, figure pipelines, and the Journal of Imaging manuscript sources under [`draft/`](draft/).

It is maintained independently of the original TransMorph release. The upstream implementation and paper are by Junyu Chen et al.; see **Citation** below. A copy of the original remote is kept as Git remote `upstream` for reference.

**Public URL:** [https://github.com/Zzxmh/TransMorph_Transformer_for_Medical_Image_Registration](https://github.com/Zzxmh/TransMorph_Transformer_for_Medical_Image_Registration)

---

## What is in this repository

| Path | Purpose |
|------|---------|
| [`IXI/`](IXI/) | Unified evaluation (`eval_any.py`, `run_full_eval.py`), metrics, adapters, HypEReg/TransMorph training code under `IXI/TransMorph/`, baseline code under `IXI/Baseline_*`, cached eval outputs under `IXI/Eval_Results/` (optional locally). |
| [`OASIS/`](OASIS/) | Optional OASIS-oriented scripts (deferred in the current manuscript). |
| [`draft/`](draft/) | LaTeX manuscript (`template_ixi_only.tex`), bibliography, figures used in the paper. |
| [`docs/`](docs/) | Supplementary index, MDPI checklist, release/cleanup notes, file classification. |
| [`figures/`](figures/) | Figure regeneration script(s), e.g. `regenerate_figures.py`. |
| [`Docker/`](Docker/) | Optional container workflows. |
| [`PreprocessingMRI.md`](PreprocessingMRI.md) | Notes on brain MRI preprocessing (e.g. FreeSurfer-oriented workflow). |

Large binaries (checkpoints, full NIfTI datasets) are **not** required to be in Git; use local paths or release assets and keep checksums. See [`.gitignore`](.gitignore).

---

## Repository environment

Use a **virtual environment** at the **repository root** (the directory that contains `IXI/`, `IXI_data/`, `draft/`, etc.).

**Python:** 3.8+ is typically sufficient; match the PyTorch wheel you install.

**1) Create and activate a venv (recommended)**

```bash
cd /path/to/TransMorph_Transformer_for_Medical_Image_Registration
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

**2) Install dependencies**

Baseline + manuscript utilities:

```bash
pip install -r requirements.txt
```

For **submission-style pins** (preferred when reproducing numbers):

```bash
pip install -r requirements-stable.txt
```

For the **full IXI evaluation stack** (YAML, SciPy, plotting, ANTs/SimpleITK stack used by some baselines):

```bash
pip install -r IXI/requirements-ixi-eval.txt
```

**PyTorch / CUDA:** install builds appropriate to your GPU and driver from the [PyTorch install guide](https://pytorch.org/get-started/locally/). The repo root `requirements.txt` lists `torch` and `torchvision` without pinning a CUDA index.

**GPU:** Training scripts under [`IXI/TransMorph/`](IXI/TransMorph/) call `.cuda()` and expect a CUDA device unless you modify them. The unified evaluator [`IXI/eval_any.py`](IXI/eval_any.py) can fall back to CPU for adapters that support it, but full benchmarks are normally run on GPU.

---

## Data layout

- Place or symlink preprocessed IXI data under [`IXI_data/`](IXI_data/) at the **repository root** with at least:
  - `atlas.pkl` (or `altas.pkl`)
  - `Train/`, `Val/`, `Test/` folders of per-subject `.pkl` pairs (as used by `IXI/TransMorph` datasets)
- Paths in [`IXI/eval_configs.yaml`](IXI/eval_configs.yaml) (`ixi_root`, `test_subdir`, `atlas`) must match your layout.
- Segmentation and preprocessing conventions: [`PreprocessingMRI.md`](PreprocessingMRI.md).
- Optional inventory: [`docs/IXI_OASIS_FILE_CLASSIFICATION.md`](docs/IXI_OASIS_FILE_CLASSIFICATION.md).

---

## Training (IXI, TransMorph family)

Training scripts live in [`IXI/TransMorph/`](IXI/TransMorph/). They resolve `IXI_data` as **two levels up** from that folder (repo root) and write **`experiments/`** and **`logs/`** relative to the **current working directory**.

**Always run training from `IXI/TransMorph`:**

```bash
cd IXI/TransMorph
```

| Command | Model / objective |
|--------|-------------------|
| `python train_TransMorph_her.py` | HypEReg-TransMorph (NCC + Grad3d + hyperelastic regularizer, HypEReg with `alpha_length=0`) |
| `python train_TransMorph.py` | TransMorph (NCC + Grad3d, manuscript baseline) |
| `python train_TransMorph_Bayes.py` | TransMorph-Bayes |
| `python train_TransMorph_diff.py` | TransMorph-diff |
| `python train_TransMorph_bspl.py` | TransMorph-bspl |

Hyperparameters (learning rate, epochs, loss weights, `cont_training`, experiment folder names, etc.) are set **inside each script**; edit the file to change training duration or loss weights. Checkpoints appear under `IXI/TransMorph/experiments/<run_name>/`.

---

## Inference and evaluation

### A) Unified IXI pipeline (recommended)

From the **repository root**, configure models in [`IXI/eval_configs.yaml`](IXI/eval_configs.yaml) (each entry: `id`, `adapter`, checkpoint paths as needed).

**Full metrics on the test set** (writes `IXI/Eval_Results/<model_id>/` with `per_case.csv`, `meta.json`, etc.):

```bash
python IXI/run_full_eval.py --config IXI/eval_configs.yaml --only transmorph_her,transmorph_original
```

Useful flags:

```bash
python IXI/run_full_eval.py --config IXI/eval_configs.yaml --only midir --max-cases 5 --no-aggregate
```

- `--only` &mdash; comma-separated model ids from the YAML
- `--skip` &mdash; skip listed ids
- `--max-cases` &mdash; limit cases (debug)
- `--no-aggregate` &mdash; skip final aggregation step

**Lightweight pass** (mean Dice + `non_jec` only; good for new uploaded weights):

```bash
python IXI/eval_light_uploaded.py --config IXI/eval_configs.yaml --models transmorph_original,transmorphbayes --out_dir IXI/Results/uploaded_weights_light
```

**Inferential tables** (Wilcoxon + Benjamini&ndash;Hochberg):

```bash
python -m IXI.analysis_comprehensive.export_inferential_stats --eval_root IXI/Eval_Results --out_csv IXI/Results/comprehensive/sig_matrix_extended.csv
python IXI/analysis_comprehensive/export_uploaded_weight_stats.py --eval_root IXI/Eval_Results --light_root IXI/Results/uploaded_weights_light --out_csv IXI/Results/comprehensive/sig_matrix_uploaded_ckpt.csv
```

**Pure forward runtime / peak GPU memory** (isolated adapter forward pass):

```bash
python IXI/profile_forward_runtime_memory.py --subject subject_1.pkl --repeats 20
```

### B) Legacy `IXI/TransMorph` infer scripts

These write CSVs under `IXI/TransMorph/Quantitative_Results/` and assume you run from **`IXI/TransMorph`**:

```bash
cd IXI/TransMorph
python infer_TransMorph_her.py
```

`infer_TransMorph_her.py` expects the HypEReg training run folder and a checkpoint name as coded in that file (default HypEReg experiment directory and `dsc0.743.pth.tar`-style naming).

For `infer_TransMorph.py` (and other `infer_*.py` variants), edit **inside the script** the placeholders such as `atlas_dir`, `test_dir`, and `model_folder` so they point to your `IXI_data` split and trained `experiments/` directory.

---

## Manuscript figures

From the **repository root** (requires `IXI_data` and result CSVs as expected by the script):

```bash
python figures/regenerate_figures.py --fig_dir figures --subject subject_1.pkl --models transmorph_her,transmorph_original,midir
```

Further command-to-output mapping and release checklist: [`README_RELEASE.md`](README_RELEASE.md).

---

## Manuscript and supplements

- Main LaTeX: [`draft/template_ixi_only.tex`](draft/template_ixi_only.tex)
- Supplementary / journal compliance: [`docs/SUPPLEMENTARY_MATERIALS_INDEX.md`](docs/SUPPLEMENTARY_MATERIALS_INDEX.md), [`docs/MDPI_JIMAGING_COMPLIANCE_CHECKLIST.md`](docs/MDPI_JIMAGING_COMPLIANCE_CHECKLIST.md)

---

## Upstream Credit and License Compliance

- This repository is a derivative research codebase built on top of the original **TransMorph** project by **Junyu Chen et al.**.
- Upstream repository: [junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
- Upstream paper: [TransMorph: Transformer for Unsupervised Medical Image Registration (MedIA 2022)](https://www.sciencedirect.com/science/article/pii/S1361841522002432)
- We keep attribution in documentation and manuscript text, and retain the upstream citation whenever the backbone or inherited components are used.
- License obligations for redistribution are summarized in [`NOTICE`](NOTICE) together with attribution guidance.

---

## Related work (TransMorph)

The TransMorph architecture and original codebase are due to Junyu Chen et al. If you use the **original** TransMorph method or their baselines, cite:

- Paper: [TransMorph: Transformer for Unsupervised Medical Image Registration (MedIA 2022)](https://www.sciencedirect.com/science/article/pii/S1361841522002432)
- arXiv: [2111.10480](https://arxiv.org/abs/2111.10480)

```bibtex
@article{chen2022transmorph,
  title = {TransMorph: Transformer for unsupervised medical image registration},
  journal = {Medical Image Analysis},
  pages = {102615},
  year = {2022},
  issn = {1361-8415},
  doi = {10.1016/j.media.2022.102615},
  url = {https://www.sciencedirect.com/science/article/pii/S1361841522002432},
  author = {Junyu Chen and Eric C. Frey and Yufan He and William P. Segars and Ye Li and Yong Du}
}
```

When HypEReg-TransMorph or this repository&rsquo;s evaluation protocols are relevant, also cite your **Journal of Imaging** article (or preprint) once available, and retain the TransMorph citation above where the backbone applies.

---

## License

See [`LICENSE`](LICENSE) (MIT; copyright Mohan Xu and Shiyi Xu for this project distribution; third-party and upstream code may retain its own notices/licenses in subdirectories). See [`NOTICE`](NOTICE) for explicit upstream credit and citation guidance.
