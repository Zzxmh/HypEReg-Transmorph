# IXI / OASIS File Classification (Manuscript Contribution Audit)

Scope: HER-TransMorph manuscript is currently IXI-only. OASIS work is deferred.

Labels:
- KEEP_MANUSCRIPT: directly contributes to the current paper (referenced or used to produce reported numbers).
- KEEP_INFRA: infrastructure used by KEEP_MANUSCRIPT scripts.
- REDUNDANT: inherited upstream-only code or local-only utilities, not used by the paper.
- DELETE_NOW: strictly redundant or per-user instruction; safe to remove.
- DEFERRED: kept only because future work may reuse (OASIS).

## A) IXI top-level

| Path | Label | Reason |
|---|---|---|
| `IXI/__init__.py` | KEEP_INFRA | package marker |
| `IXI/eval_any.py` | KEEP_MANUSCRIPT | core per-pair inference + metric pipeline |
| `IXI/eval_configs.yaml` | KEEP_MANUSCRIPT | cited in Methods/Appendix |
| `IXI/eval_light_uploaded.py` | KEEP_MANUSCRIPT | uploaded checkpoint paired analysis |
| `IXI/metrics_full.py` | KEEP_MANUSCRIPT | metric implementations |
| `IXI/profile_forward_runtime_memory.py` | KEEP_MANUSCRIPT | cited in Section 4.3/Appendix A.1 |
| `IXI/run_full_eval.py` | KEEP_MANUSCRIPT | main eval runner cited in Appendix |
| `IXI/aggregate_and_plot.py` | KEEP_INFRA | used by `run_full_eval.py` |
| `IXI/augment_legacy_csv.py` | KEEP_INFRA | used by `run_full_eval.py` |
| `IXI/analysis.py` | KEEP_MANUSCRIPT | cited for grouped Dice protocol |
| `IXI/analysis_trans.py` | KEEP_MANUSCRIPT | cited for grouped Dice protocol |
| `IXI/requirements-ixi-eval.txt` | KEEP_INFRA | environment file |
| `IXI/Anatomical_Structures.md` | DELETE_NOW | inherited from original TransMorph repo |
| `IXI/TransMorph_on_IXI.md` | DELETE_NOW | inherited from original repo (per user instruction) |
| `IXI/配准评估指标系统性综述.md` | DELETE_NOW | internal Chinese metric notes |
| `IXI/eval_configs_reverse.yaml` | DELETE_NOW | reverse-eval, not in manuscript |
| `IXI/print_model_grid.py` | DELETE_NOW | local utility, not in manuscript |
| `IXI/monitor_her_eval.ps1` | DELETE_NOW | personal automation |
| `IXI/smoke_test_full_eval.py` | REDUNDANT | smoke test (optional to keep) |

## B) IXI/adapters

| Path | Label | Reason |
|---|---|---|
| `IXI/adapters/__init__.py` | KEEP_INFRA | package marker |
| `IXI/adapters/_helpers.py` | KEEP_INFRA | shared helpers |
| `IXI/adapters/transmorph_her.py` | KEEP_MANUSCRIPT | HER-TransMorph adapter |
| `IXI/adapters/transmorph_original.py` | KEEP_MANUSCRIPT | TransMorph baseline adapter |
| `IXI/adapters/transmorphbayes.py` | KEEP_MANUSCRIPT | TransMorphBayes adapter (uploaded ckpt) |
| `IXI/adapters/voxelmorph_1.py` | KEEP_MANUSCRIPT | VoxelMorph-1 baseline |
| `IXI/adapters/cyclemorph.py` | KEEP_MANUSCRIPT | CycleMorph baseline |
| `IXI/adapters/midir.py` | KEEP_MANUSCRIPT | MIDIR baseline |
| `IXI/adapters/cotr.py` | KEEP_MANUSCRIPT | CoTr baseline |
| `IXI/adapters/nnformer.py` | KEEP_MANUSCRIPT | nnFormer baseline |
| `IXI/adapters/pvt.py` | KEEP_MANUSCRIPT | PVT baseline |
| `IXI/adapters/syn.py` | KEEP_MANUSCRIPT | classical SyN baseline |
| `IXI/adapters/syn_legacy_to_per_case.py` | KEEP_INFRA | legacy SyN compatibility |
| `IXI/adapters/affine.py` | DELETE_NOW | affine baseline removed from manuscript |
| `IXI/adapters/transmorph_psc.py` | DELETE_NOW | PSC variant not in manuscript |

## C) IXI/analysis_comprehensive

| Path | Label | Reason |
|---|---|---|
| `analysis_comprehensive/__init__.py` | KEEP_INFRA | package marker |
| `analysis_comprehensive/__main__.py` | KEEP_INFRA | CLI entry |
| `analysis_comprehensive/build_table.py` | KEEP_MANUSCRIPT | builds reported tables |
| `analysis_comprehensive/config.py` | KEEP_INFRA | model registry |
| `analysis_comprehensive/dispatch.py` | KEEP_INFRA | task dispatcher |
| `analysis_comprehensive/metrics.py` | KEEP_INFRA | metric helpers |
| `analysis_comprehensive/plot_all.py` | KEEP_MANUSCRIPT | aggregate plots |
| `analysis_comprehensive/run_inference.py` | KEEP_MANUSCRIPT | per-model inference |
| `analysis_comprehensive/stats.py` | KEEP_MANUSCRIPT | core Wilcoxon + BH-FDR |
| `analysis_comprehensive/export_inferential_stats.py` | KEEP_MANUSCRIPT | Table S2 source |
| `analysis_comprehensive/export_uploaded_weight_stats.py` | KEEP_MANUSCRIPT | uploaded ckpt stats |
| `analysis_comprehensive/generate_submission_figures.py` | KEEP_MANUSCRIPT | submission figure generator |
| `analysis_comprehensive/merge_baseline_table.py` | KEEP_INFRA | baseline merge |
| `analysis_comprehensive/auto_refresh_interim.py` | DELETE_NOW | interim workflow only |
| `analysis_comprehensive/interim_completed_report.py` | DELETE_NOW | interim workflow only |

## D) IXI/TransMorph (model package)

| Path | Label | Reason |
|---|---|---|
| `IXI/TransMorph/train_TransMorph_her.py` | KEEP_MANUSCRIPT | HER training |
| `IXI/TransMorph/infer_TransMorph_her.py` | KEEP_MANUSCRIPT | HER inference reference |
| `IXI/TransMorph/train_TransMorph.py` | KEEP_MANUSCRIPT | TransMorph baseline retraining |
| `IXI/TransMorph/losses.py` | KEEP_INFRA | imported by training scripts |
| `IXI/TransMorph/losses_her.py` | KEEP_MANUSCRIPT | HER loss implementation |
| `IXI/TransMorph/utils.py` | KEEP_INFRA | `process_label`, `register_model` etc. |
| `IXI/TransMorph/label_info.txt` | KEEP_INFRA | label table for `process_label` |
| `IXI/TransMorph/voi_definitions.py` | KEEP_INFRA | VOI definitions |
| `IXI/TransMorph/data/datasets.py`, `data_utils.py`, `rand.py`, `trans.py`, `__init__.py` | KEEP_INFRA | dataset loader |
| `IXI/TransMorph/models/configs_TransMorph.py`, `TransMorph.py`, `__init__.py` | KEEP_INFRA | TransMorph backbone |
| `IXI/TransMorph/models/configs_TransMorph_Bayes.py`, `TransMorph_Bayes.py` | KEEP_INFRA | required by Bayes adapter |
| `IXI/TransMorph/models/configs_TransMorph_bspl.py`, `TransMorph_bspl.py` | DELETE_NOW | bspl variant unused |
| `IXI/TransMorph/models/configs_TransMorph_diff.py`, `TransMorph_diff.py` | DELETE_NOW | diff variant unused |
| `IXI/TransMorph/models/finite_differences.py` | DELETE_NOW | only used by diff variant |
| `IXI/TransMorph/models/transformation.py` | DELETE_NOW | unused in IXI evaluation pipeline |
| `IXI/TransMorph/infer_TransMorph.py` | DELETE_NOW | original-repo inference, replaced by adapter |
| `IXI/TransMorph/infer_TransMorph_Bayes.py` | DELETE_NOW | replaced by adapter |
| `IXI/TransMorph/infer_TransMorph_bspl.py` | DELETE_NOW | bspl variant unused |
| `IXI/TransMorph/infer_TransMorph_diff.py` | DELETE_NOW | diff variant unused |
| `IXI/TransMorph/train_TransMorph_Bayes.py` | DELETE_NOW | not retrained in this manuscript |
| `IXI/TransMorph/train_TransMorph_bspl.py` | DELETE_NOW | bspl variant unused |
| `IXI/TransMorph/train_TransMorph_diff.py` | DELETE_NOW | diff variant unused |
| `IXI/TransMorph/run_train_*.cmd`, `run_train_*.ps1` | DELETE_NOW | personal Windows automation |
| `IXI/TransMorph/experiments/` | KEEP_INFRA (gitignored) | local checkpoints |
| `IXI/TransMorph/logs/` | DELETE_NOW (gitignored) | runtime logs |
| `IXI/TransMorph/Quantitative_Results/` | DELETE_NOW (gitignored) | intermediate CSVs |

## E) IXI/Baseline_registration_methods

Baselines below are evaluated through unified adapters; per-baseline `infer.py` and most legacy scripts are inherited from the upstream TransMorph repo.

| Path | Label | Reason |
|---|---|---|
| `Baseline_registration_methods/VoxelMorph/train_vxm.py` | KEEP_MANUSCRIPT | retraining-capable script |
| `Baseline_registration_methods/VoxelMorph/{models.py,losses.py,utils.py,label_info.txt,data/}` | KEEP_INFRA | required for VoxelMorph adapter loading |
| `Baseline_registration_methods/VoxelMorph/VoxelMorph_1_Validation_dsc0.720.pth.tar` | KEEP_INFRA (gitignored) | local weight |
| `Baseline_registration_methods/VoxelMorph/{experiments,logs}/` | DELETE_NOW (gitignored) | local outputs |
| `Baseline_registration_methods/VoxelMorph/infer.py` | DELETE_NOW | replaced by adapter |
| `Baseline_registration_methods/MIDIR/train_MIDIR.py` | KEEP_MANUSCRIPT | retraining-capable script |
| `Baseline_registration_methods/MIDIR/{models.py,transformation.py,losses.py,utils.py,label_info.txt,data/}` | KEEP_INFRA | required for MIDIR adapter |
| `Baseline_registration_methods/MIDIR/MIDIR_Validation_dsc0.733.pth.tar` | KEEP_INFRA (gitignored) | local weight |
| `Baseline_registration_methods/MIDIR/infer.py` | DELETE_NOW | replaced by adapter |
| `Baseline_registration_methods/CycleMorph/train.py`, `infer.py` | DELETE_NOW | not retrained, replaced by adapter |
| `Baseline_registration_methods/CycleMorph/{models/,losses.py,utils.py,util/,label_info.txt,data/}` | KEEP_INFRA | required for CycleMorph adapter |
| `Baseline_registration_methods/CycleMorph/CycleMorph_Validation_dsc0.729.pth.tar` | KEEP_INFRA (gitignored) | local weight |
| `Baseline_registration_methods/VoxelMorph-diff/` | DELETE_NOW | not in manuscript baseline list |
| `Baseline_registration_methods/runner_logs/` | DELETE_NOW (gitignored) | runtime logs |
| `Baseline_registration_methods/run_vxm_then_midir.ps1` | DELETE_NOW | personal automation |

## F) IXI/Baseline_Transformers

| Path | Label | Reason |
|---|---|---|
| `Baseline_Transformers/models/` | KEEP_INFRA | required for CoTr/nnFormer/PVT adapters |
| `Baseline_Transformers/data/` | KEEP_INFRA | dataset loaders for adapters |
| `Baseline_Transformers/{utils.py,losses.py,label_info.txt}` | KEEP_INFRA | shared utilities |
| `Baseline_Transformers/infer_{CoTr,nnFormer,PVT}.py` | DELETE_NOW | replaced by adapters |
| `Baseline_Transformers/train_{CoTr,nnFormer,PVT}.py` | DELETE_NOW | not retrained in this manuscript |
| `Baseline_Transformers/infer_ViTVNet.py`, `train_ViTVNet.py` | DELETE_NOW | ViT-V-Net not in manuscript baselines |
| `Baseline_Transformers/PVT_Validation_dsc0.720.pth.tar` | KEEP_INFRA (gitignored) | local weight |

## G) IXI/Baseline_traditional_methods

| Path | Label | Reason |
|---|---|---|
| `Baseline_traditional_methods/SyN/` | DELETE_NOW | manuscript uses ANTsPy adapter |
| `Baseline_traditional_methods/deedsBCV/` | DELETE_NOW | not in manuscript baselines |
| `Baseline_traditional_methods/LDDMM/` | DELETE_NOW | not in manuscript baselines |
| `Baseline_traditional_methods/NiftyReg/` | DELETE_NOW | not in manuscript baselines |

## H) IXI/Eval_Results, IXI/Results, IXI/Eval_Results_reverse, IXI/IXI_data

| Path | Label | Reason |
|---|---|---|
| `IXI/Eval_Results/` | KEEP_INFRA (gitignored) | per-model per-case CSVs (regenerable) |
| `IXI/Results/comprehensive/` | KEEP_MANUSCRIPT | inferential CSVs cited in manuscript |
| `IXI/Results/uploaded_weights_light/` | KEEP_MANUSCRIPT (gitignored) | uploaded ckpt per-case |
| `IXI/Results/comprehensive_bayes_eval/` | DELETE_NOW (gitignored) | superseded by uploaded ckpt analysis |
| `IXI/Results/*.csv` (legacy upstream CSVs) | KEEP_INFRA | grouped Dice source for `analysis.py`/`analysis_trans.py` |
| `IXI/Eval_Results_reverse/` | DELETE_NOW | reverse-eval not in manuscript |
| `IXI/Eval_Results/her_eval_*.log`, `live.log` | DELETE_NOW (gitignored) | runtime logs |

## I) OASIS folder (manuscript IXI-only; treat as DEFERRED)

OASIS is **not contributing** to the current manuscript (deferred). Two options:
- DEFERRED retain (recommended if follow-up work expected),
- DELETE_NOW (recommended only if sure the project will not extend to OASIS).

| Path | Label | Reason |
|---|---|---|
| `OASIS/TransMorph_on_OASIS.md` | DELETE_NOW | inherited from original repo (per user instruction) |
| `OASIS/README.md`, `subjects.txt`, `seg*_labels.txt` | DEFERRED | inherited Learn2Reg metadata |
| `OASIS/evaluation.py` | DEFERRED | inherited evaluation utility |
| `OASIS/eval_configs.yaml`, `eval_oasis.py` | DEFERRED | our cross-dataset extension scaffolding |
| `OASIS/monitor_train_her.ps1` | DELETE_NOW | personal automation |
| `OASIS/data/` | DEFERRED (gitignored) | local dataset |
| `OASIS/Eval_Results/` | DEFERRED (gitignored) | empty/_stats only |
| `OASIS/surface_distance/` | DEFERRED | metric implementation reusable |
| `OASIS/TransMorph/{train_TransMorph_her,train_TransMorph_unsup,submit_TransMorph_her,losses_her}.py` | DEFERRED | our HER scaffolding |
| `OASIS/TransMorph/{losses,utils,train_TransMorph,submit_TransMorph}.py` | DEFERRED | inherited base |
| `OASIS/TransMorph/{data,models,experiments,logs}/` | DEFERRED | required by HER training |
| `OASIS/TransMorph/TransMorph_Validation_dsc0.857.pth.tar` | KEEP_INFRA (gitignored) | local weight |

## J) Mandatory Deletions (this round)

The following are deleted now per user instruction and clear redundancy:

- `IXI/TransMorph_on_IXI.md`
- `OASIS/TransMorph_on_OASIS.md`
- `IXI/Anatomical_Structures.md`
- `IXI/配准评估指标系统性综述.md`
- `IXI/eval_configs_reverse.yaml`
- `IXI/print_model_grid.py`
- `IXI/monitor_her_eval.ps1`
- `IXI/Eval_Results_reverse/` (entire subtree, gitignored already; remove from working tree)

Optional deletions (recommended but not enforced in this round):
- `IXI/adapters/affine.py`, `IXI/adapters/transmorph_psc.py`
- `IXI/analysis_comprehensive/auto_refresh_interim.py`, `interim_completed_report.py`
- bspl/diff variants under `IXI/TransMorph/{infer,train,models}/*`
- `IXI/TransMorph/run_train_*.cmd`, `run_train_*.ps1`
- `IXI/Baseline_registration_methods/VoxelMorph-diff/`
- `IXI/Baseline_registration_methods/run_vxm_then_midir.ps1`
- `IXI/Baseline_registration_methods/runner_logs/`
- `IXI/Baseline_traditional_methods/{deedsBCV,LDDMM,NiftyReg,SyN}/`
- `IXI/Baseline_Transformers/infer_*ViTVNet*.py`, `train_*ViTVNet*.py`
- `IXI/Baseline_Transformers/infer_{CoTr,nnFormer,PVT}.py`, `train_{CoTr,nnFormer,PVT}.py`
- `IXI/smoke_test_full_eval.py`
- `OASIS/monitor_train_her.ps1`

These optional deletions should only be performed when the user explicitly approves them.
