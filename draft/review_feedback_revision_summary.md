# Revision Summary for `template_ixi_only`

This document summarizes all revisions executed in response to the review feedback, plus remaining items that still require additional experiments, statistics exports, or public-release artifacts before a fully submission-ready version.

## Completed Revisions

### 1) Title, abstract, and claim calibration
- Updated title to a safer claim: **folding-suppressed / hyperelastic-regularized** wording (removed "topology-preserving" from title).
- Rewrote abstract for numerical consistency with the current Table 2 values:
  - Dice: `0.7537 ± 0.0275`
  - non-positive Jacobian determinant ratio: `0.000015 ± 0.000007` (HER) vs `0.015634 ± 0.003363` (TransMorphBayes)
  - SDlogJ: `0.3280 ± 0.0221` (HER) vs `0.4920 ± 0.0330` (TransMorphBayes)
- Defined key abbreviations in abstract context (HER, IXI, SDlogJ wording).
- Replaced over-strong "topology-preserving" phrasing with "Jacobian-regularized" / "folding-suppressed" across key narrative sections.

### 2) Introduction and related-work framing
- Removed journal-targeting sentence from Introduction.
- Reframed contributions to be evidence-based and method-accurate (regularization strategy on TransMorph-style backbone, not a new architecture).
- Restructured Related Work into four reviewer-requested buckets:
  - Classical and diffeomorphic registration
  - Learning-based dense registration
  - Transformer-based registration
  - Hyperelastic/Jacobian regularization

### 3) Dataset and preprocessing reproducibility detail
- Replaced vague preprocessing wording ("repository pipeline") with explicit protocol language.
- Expanded the dataset/protocol summary table to include:
  - data source
  - license statement
  - volume counts and split
  - modality/task
  - preprocessing summary
  - labels and grouped-VOI protocol
  - atlas definition
  - interpolation policy
  - spacing-aware metric computation note

### 4) Method and loss formulation consistency
- Unified loss notation and equations:
  - single total-loss expression with `L_sim`, `L_grad`, and HER weighted sub-terms
  - explicit `phi(x)=x+u(x)` transform definition
- Clarified `L_sim` as **negative LNCC loss**.
- Added Jacobian implementation detail (forward finite differences on interior stencil) aligned to current HER loss implementation.
- Added behavior explanation for `det(J_phi) <= 0` under clamp + folding penalty.
- Added an editable **Algorithm 1** training-step block in text form (not image).

### 5) Training details, fairness, and baseline provenance
- Expanded training details with optimizer flavor (AMSGrad), weight decay, checkpoint selection criterion, and seed limitation disclosure.
- Added a new **Baseline reproducibility and provenance table** covering:
  - source/checkpoint provenance
  - whether trained by authors
  - test-split exposure verification status
  - inference implementation notes
- Included explicit SyN adapter settings currently used (`SyNOnly`, `meansquares`, `(160,80,40)` iterations).

### 6) Metrics/statistics wording and notation fixes
- Corrected Jacobian-ratio notation globally:
  - from invalid `|J| < 0` style
  - to `det(J_phi) <= 0` ratio
- Added metric-definition details for direction and units (HD95/ASSD in mm, SDlogJ clipping form).
- Rewrote statistical-analysis paragraph to avoid unsupported completed-significance claims in the current IXI-only manuscript state.

### 7) Results consistency and interpretation
- Removed placeholder/pending statistical references from Results text.
- Corrected regularity numbers in prose to match current table values and comparison targets.
- Rewrote efficiency interpretation:
  - removed incorrect causal claim that HER training loss explains inference-time difference
  - clarified implementation/evaluation-path variance vs architecture-level equivalence

### 8) Tables and figure captions
- Updated Table 2 headers and units:
  - `det(J_phi) <= 0 ratio`
  - `HD95 (mm)`
  - `ASSD (mm)`
- Added best/second-best highlighting in Table 2 (bold/underline) and standardized note as mean ± SD.
- Updated Table 3 SyN parameters from `0` to `N/A` where not applicable.
- Revised figure captions for consistent technical language:
  - Figure 2 naming consistency and purpose clarity
  - Figure 3 changed from "topology-preserving" wording to folding-risk interpretation
  - Figure 4 changed `log(J)` wording to `log(det(J_phi))`
  - Figure 5 caption cleaned (no local-workflow phrasing)

### 9) Manuscript hygiene and compliance-related cleanup
- Removed all `Stage-*`, `pending`, `TBD`, and `\PH{...}` placeholders from `template_ixi_only.tex`.
- Rewrote supplementary and data-availability text to remove submission-blocking placeholder language.
- Updated Appendix extended table (`tab5`) to descriptive reporting only (removed pending q-value column entries).

### 10) Build verification
- Recompiled `draft/template_ixi_only.tex` successfully after edits.
- Confirmed Figure 1 still loads from `draft/HER.png`.

---

## Remaining Necessities for Full Submission Completeness

These items were **not fully executable in text-only revision** and still require new data export, reruns, or external release actions:

1. **Final inferential statistics package**
   - Complete subject-level paired Wilcoxon + BH-FDR outputs for all reported metric families.
   - Add final `p`/`q` tables and effect sizes in manuscript/supplement.

2. **Figure 5 redesign at source level**
   - Replace Roman-numeral x-axis labels in the rendered figure file with readable model names.
   - Add uncertainty/error bars (e.g., CI/SE) if intended for main-text evidence.
   - Re-render figure asset accordingly.

3. **Public reproducibility release**
   - Publish versioned code repository and archive DOI (e.g., Zenodo/OSF).
   - Include frozen commit hash, environment lockfile/container, split manifests, checkpoint hashes, per-case tables, and figure/stat scripts in public package.
   - Replace temporary review-package wording in Data Availability with final persistent links.

4. **Fairness-strengthening experiments (if required by target reviewers/editors)**
   - Retraining or independently verified split provenance for key baseline checkpoints.
   - Optional additions: topology-focused comparators (e.g., diffeomorphic/BSpline variants under same split) if required for stronger claims.

5. **Sensitivity and robustness**
   - Multi-seed experiments and confidence intervals.
   - Beta/gamma sensitivity sweep for HER coefficients.
   - Optional expanded ablations (`volume-only`, `fold-only`, combined) under final pipeline.

6. **Final formatting polish**
   - Current PDF still compiles with non-fatal layout warnings (mainly table width/headheight). Submission polish pass should resolve those warnings for cleaner production output.

---

## Files Modified

- `draft/template_ixi_only.tex`
- `draft/review_feedback_revision_summary.md` (this report)

---

## Additional Pass (Latest Reviewer Round)

The following additional revisions were applied after a subsequent detailed review:

- Further claim calibration in Results/Conclusions:
  - conclusions now explicitly state improvements vs unconstrained dense Transformer baselines,
  - and explicitly acknowledge MIDIR/SyN strength on selected regularity metrics.
- Figure 1 loss notation consistency fixed in caption:
  - changed from `L_NCC` wording to `L_sim`, with explicit `L_sim = -LNCC`.
- Figure 5 moved out of main-text evidence path:
  - removed from the core Results narrative,
  - retained as supplementary diagnostic panel (`Figure S1`) only.
- Qualitative-claim wording softened:
  - changed “demonstrates” to “qualitatively suggests” where appropriate.
- Baseline provenance table simplified for readability in the main text.
- Dataset preprocessing wording made more explicit (T1 preprocessed IXI workflow; skull stripping / affine alignment / segmentation preprocessing references).
- Data-availability, IRB, and informed-consent wording tightened for safer submission phrasing.
- Appendix table with ambiguous “Best Baseline” logic replaced by model-wise comparative columns (`HER`, `TransMorphBayes`, `MIDIR`) for clearer interpretation.

### Still Pending (cannot be completed by text editing alone)

- Public reproducibility release with final repository URL + archival DOI.
- Multi-seed and sensitivity studies (beta/gamma) if required for final claims.

### Newly Completed in This Execution Round

- Generated full inferential statistics export with paired median differences, BH-FDR q-values, and signed effect sizes:
  - `IXI/Results/comprehensive/sig_matrix_extended.csv`
- Recomputed summary/significance exports from available per-case outputs:
  - `IXI/Results/comprehensive/model_summary.csv`
  - `IXI/Results/comprehensive/sig_matrix.csv`
- Added manuscript-integrated inferential reporting:
  - updated statistical-method text,
  - added Supplementary Table S2 in `template_ixi_only.tex`,
  - integrated uploaded-checkpoint inferential rows for TransMorph/TransMorphBayes (Dice + non-positive Jacobian ratio).
- Regenerated figure assets from source script after redesign:
  - Figure labels now use readable model names (no Roman numerals),
  - Figure 5 now includes error bars and corrected `det(J_phi) <= 0` notation,
  - Jacobian histogram axis text updated to `log det(J_phi)`.
- Added uploaded-checkpoint evaluation and statistics pipeline:
  - `IXI/adapters/transmorph_original.py` now prioritizes uploaded checkpoint `TransMorph_Validation_dsc0.744.pth.tar`.
  - Added `IXI/adapters/transmorphbayes.py` and `IXI/eval_configs.yaml` entry `transmorphbayes`.
  - Added `IXI/eval_light_uploaded.py` and ran full 115-case light evaluation for:
    - `IXI/Results/uploaded_weights_light/transmorph_original_light_per_case.csv`
    - `IXI/Results/uploaded_weights_light/transmorphbayes_light_per_case.csv`
  - Added `IXI/analysis_comprehensive/export_uploaded_weight_stats.py` and exported:
    - `IXI/Results/comprehensive/sig_matrix_uploaded_ckpt.csv`
  - Updated Supplementary Table S2 / appendix note in `template_ixi_only.tex` to include uploaded-checkpoint inferential results.

### Newly Completed for Round-2 Minor Revision Requirements

- Implemented low-cost/high-impact P0 textual and reproducibility fixes in `template_ixi_only.tex`:
  - Title explicitly scoped to IXI atlas-to-subject protocol.
  - `alpha=0` framing clarified as the volume + anti-folding hyperelastic subset.
  - Equation-context wording and bidirectional optimization protocol clarified.
  - Added explicit grouped-VOI protocol subsection and a new Supplementary Table S3 (grouped VOI to label IDs).
  - Added explicit metric implementation details (LNCC window, NMI bins, SSIM range/window policy).
  - Added explicit signed-effect definition as matched-pairs rank-biserial correlation.
  - Added numerical explanation note for repeated ultra-small Wilcoxon p-values (floating-point practical floor behavior).
- Added citation-completeness fixes in `draft/refs.bib` and manuscript citations:
  - Added AMSGrad (Reddi et al.), STN (Jaderberg et al.), probabilistic diffeomorphic VoxelMorph (Dalca et al.), NMI (Studholme et al.), SSIM (Wang et al.), and Bookstein TPS references.
  - Updated nnFormer citation to IEEE TIP journal publication entry.
  - Removed ViT-V-Net from active manuscript claims/citations where not evaluated in the present benchmark.
- Added explicit adapter-level description for CoTr/nnFormer/PVT registration usage (two-channel input + repository flow head under shared IXI protocol).
- Added controlled forward-only profiling script:
  - `IXI/profile_forward_runtime_memory.py`
  - Results integrated into Results/Appendix to separate pure forward cost from end-to-end evaluation-path timing.
- Updated inferential export scripts to use rank-biserial signed effect:
  - `IXI/analysis_comprehensive/export_inferential_stats.py`
  - `IXI/analysis_comprehensive/export_uploaded_weight_stats.py`
  - Re-exported `sig_matrix_extended.csv` and `sig_matrix_uploaded_ckpt.csv`, and synchronized Table S2 values.

