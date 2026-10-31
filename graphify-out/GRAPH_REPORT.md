# Graph Report - mp-factory  (2026-10-30)

## Corpus Check
- 101 files · ~1,784,292 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 76 file(s) not represented in the graph (top: .sbatch 67, .nrrd 6, (none) 2)

## Summary
- 961 nodes · 2072 edges · 69 communities (40 shown, 29 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 82 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `46674074`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- multiphase_tumor_kinetics.py
- evaluate_multi_model_ensemble.py
- json
- generate_showcase.py
- run_scaling_sweep.py
- evaluate_gi_masks.py
- test_active_learning.py
- torch
- register_multiphase_pairs.py
- os
- test_unified_gastro_dataset.py
- PART III: Master Publication Tables (Tables 1–4)
- unified_gastro_dataset.py
- evaluate_synthesis.py
- main
- sample_multiphase_diffusion.py
- MedNeXt
- quick_val_eval.py
- train_swin_unetr.py
- test_stage2_checkpoint.py
- glob
- synthesis_net.py
- stress_test_contrast_perturbation.py
- autoencoder_3d.py
- train_mednext.py
- pandas
- FixLiarLabelAffined
- _ResBlock3D
- TestPipelineDryRun
- train_mednext_phase2.py
- main
- eval_scaling_checkpoints.py
- numpy
- run_clinical_turing_test.py
- train_swin_unetr_phase2.py
- load_stage1_autoencoder
- MultiphasePairedDataset
- predict_mednext.py
- sbatch_train_mednext.sh
- train_full_anchor_4gpu.sh
- run_p10_discovery.sh
- 2. Detailed Per-Scan Clinical Annotations
- eval_option_a.sh
- eval_option_b.sh
- submit_scaling_sweep.sh
- train_full_anchor.sh
- generate_multiphase_tumors.py
- sys
- main
- predict_swin_unetr.py
- MultiPhaseTumorGenerated
- DiffusionSchedule
- discover_multiphase_cohort.py
- _center_crop_pad
- Permission & Directive: Proactive Specialized Subagents
- compute_scaling_law.py
- worst_case_outlier_audit.py
- BDMAP Radiologist Mask Reuse Notes

## God Nodes (most connected - your core abstractions)
1. `AsymmetricPDCELoss` - 18 edges
2. `_dataset()` - 16 edges
3. `mp-factory README` - 16 edges
4. `evaluate_subject()` - 15 edges
5. `main()` - 13 edges
6. `process_subject()` - 13 edges
7. `render_radiologist_comparison()` - 12 edges
8. `ActiveLearningPool` - 11 edges
9. `main()` - 11 edges
10. `UncertaintyEstimator` - 10 edges

## Surprising Connections (you probably didn't know these)
- `3. Recommended Action Plan & Implementation Steps` --references--> `AsymmetricPDCELoss`  [INFERRED]
  JHU_data_radiologist_corrected/radiologist_clinical_feedback.md → code/training/asymmetric_loss.py
- `3. Multi-Model Ensemble & Automated Data Cleansing` --references--> `triage_case()`  [INFERRED]
  CLAUDE.md → code/evaluation/evaluate_multi_model_ensemble.py
- `3. Multi-Model Ensemble & Automated Data Cleansing` --references--> `export_dataset_splits()`  [INFERRED]
  CLAUDE.md → code/evaluation/evaluate_multi_model_ensemble.py
- `6. Monitoring & Active Learning` --references--> `UncertaintyEstimator`  [INFERRED]
  CLAUDE.md → code/active_learning/loop.py
- `6. Monitoring & Active Learning` --references--> `ActiveLearningPool`  [INFERRED]
  CLAUDE.md → code/active_learning/loop.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Automated 3D Segmentation Workflows** — code_prediction_run_totalsegmentator_batch, code_slurm_scripts_run_totalseg, code_slurm_scripts_run_totalseg_bdmap, code_slurm_scripts_run_totalseg_gi_array, agents_totalsegmentator [EXTRACTED 1.00]

## Communities (69 total, 29 thin omitted)

### Community 0 - "multiphase_tumor_kinetics.py"
Cohesion: 0.13
Nodes (6): _add_texture_noise(), apply_kinetics(), _get_ring_mask(), _make_ellipsoid_mask(), _make_tumor_geometry(), PhaseEnhancement

### Community 1 - "evaluate_multi_model_ensemble.py"
Cohesion: 0.06
Nodes (24): adjusted_rand_score_fast(), compute_betti_0(), compute_classification_metrics(), compute_clustering_metrics(), compute_composite_score(), compute_diversity_weights(), compute_ece(), compute_hd95_fast() (+16 more)

### Community 2 - "json"
Cohesion: 0.28
Nodes (4): compute_subject(), main(), build_manifest(), normalise_date()

### Community 3 - "generate_showcase.py"
Cohesion: 0.07
Nodes (13): plot_curves(), apply_ct_window(), generate_comparison_figure(), generate_tumor_showcase_figure(), apply_window(), generate_interactive_3d_html(), main(), process_subject() (+5 more)

### Community 4 - "run_scaling_sweep.py"
Cohesion: 0.11
Nodes (14): MedNeXtB(), build_model(), build_transforms(), compute_dice_hd95(), _compute_hd95_numpy(), create_dummy_cohort(), evaluate_radiologist_true_dice(), _extract_3d_boundaries() (+6 more)

### Community 5 - "evaluate_gi_masks.py"
Cohesion: 0.09
Nodes (23): download.sbatch, run_evaluation.sbatch, run_totalseg.sbatch, run_totalseg_bdmap.sbatch, run_totalseg_gi_array.sbatch, smoke_test.sbatch, main.py (SyntheticTumors trainer), mp-factory README (+15 more)

### Community 6 - "test_active_learning.py"
Cohesion: 0.05
Nodes (22): 1. Data Ingestion & Management, 2. Automated 3D Segmentation, 4. Model Training (Phase 1 & Phase 2 Distillation), 5. Comparison & Expert Validation, 6. Monitoring & Active Learning, 7. Synthetic Tumor Submodule (`code/SyntheticTumors/`), AGENTS.md — Workspace Architecture & Project Guide, 📂 Codebase Directory Layout (+14 more)

### Community 7 - "torch"
Cohesion: 0.18
Nodes (8): AsymmetricPDCELoss, test_asymmetric_penalty_fn_gt_fp(), test_gradient_flows_to_backbone(), __init__(), test_ignore_index_masks_voxels(), test_ignore_index_minus_one_noop(), test_multiclass_3d_shape(), test_probability_clamping_no_nan()

### Community 8 - "register_multiphase_pairs.py"
Cohesion: 0.15
Nodes (8): affine_registration(), bspline_registration(), load_sitk_image(), main(), parse_args(), register_pair(), resample_to_fixed(), rigid_registration()

### Community 9 - "os"
Cohesion: 0.16
Nodes (10): autolabel_subject(), main(), resolve_human_label(), _require_nibabel(), test_autolabel_e2e_synthetic(), test_ignore_index_consistency(), test_organ_classes(), test_resolve_human_label_combined() (+2 more)

### Community 11 - "test_unified_gastro_dataset.py"
Cohesion: 0.24
Nodes (7): _sample_audit(), test_build_manifest_clean(), test_build_manifest_with_weak(), test_manifest_ignore_index_match(), test_manifest_no_use_weak(), test_roundtrip_dataloader(), test_verify_report_nonexistent()

### Community 12 - "PART III: Master Publication Tables (Tables 1–4)"
Cohesion: 0.06
Nodes (30): 1. Executive Summary & Clinical Scope, 1. Generative Multi-Phase Contrast CT & Anatomical Invariance Audit, 2. 4D Procedural Multi-Phase Tumor Perfusion Kinetics ($N = 1,990$ Lesions), 2. Anatomical Standards & Physical Windowing, 3. Downstream Model Segmentation Tolerance to Attenuation Shifts & Scanner Noise, 3-Plane Orthoview & JHU Radiologist Concordance, 4. Interactive 3D WebGL Viewers & Asset Downloads, 5. Conclusions & Verification Playbook (+22 more)

### Community 13 - "unified_gastro_dataset.py"
Cohesion: 0.14
Nodes (5): 3. Multi-Model Ensemble & Automated Data Cleansing, build_manifest(), main(), UnifiedGI3Dataset, verify_dataset_integrity()

### Community 14 - "evaluate_synthesis.py"
Cohesion: 0.20
Nodes (6): compute_metrics_3d(), find_matching_real_file(), load_nifti(), main(), merge_worker_csvs(), resample_real_to_target()

### Community 15 - "main"
Cohesion: 0.22
Nodes (4): main(), MultiPhaseVolumeDataset, parse_args(), setup_ddp()

### Community 16 - "sample_multiphase_diffusion.py"
Cohesion: 0.15
Nodes (4): DDIMSampler, load_anatomy_mask(), main(), preprocess_ct()

### Community 17 - "MedNeXt"
Cohesion: 0.13
Nodes (4): GRN, LayerNorm, MedNeXt, MedNeXtBlock

### Community 19 - "train_swin_unetr.py"
Cohesion: 0.16
Nodes (5): discover_dataset(), get_transforms(), GIDataset, main(), RemapGILabelTargets

### Community 20 - "test_stage2_checkpoint.py"
Cohesion: 0.12
Nodes (15): test_legacy_resume_continues_epoch_and_lr_without_new_fields(), test_legacy_resume_preserves_best_checkpoint_metric(), test_legacy_resume_requires_explicit_override(), test_main_rejects_missing_ae_before_empty_dataset(), test_new_checkpoint_records_resume_state_and_ae_identity(), test_resume_rejects_different_autoencoder_even_with_override(), test_resume_rejects_different_mask_conditioning_even_with_override(), test_resume_restores_scheduler_optimizer_and_best_loss() (+7 more)

### Community 21 - "glob"
Cohesion: 0.07
Nodes (23): BodyMaps, CancerVerse, mp-factory, SyntheticTumors, TotalSegmentator, find_ct_file(), audit_section(), check_nifti_sample() (+15 more)

### Community 23 - "stress_test_contrast_perturbation.py"
Cohesion: 0.18
Nodes (6): apply_perturbation(), compute_binary_dice(), generate_markdown_report(), main(), plot_robustness_curves(), run_model_inference()

### Community 24 - "autoencoder_3d.py"
Cohesion: 0.11
Nodes (7): AutoEncoder3DConfig, build_autoencoder(), denormalise_hu(), LightweightAutoEncoder3D, MONAI_VQVAE3D, normalise_hu(), reconstruction_loss()

### Community 25 - "train_mednext.py"
Cohesion: 0.15
Nodes (5): discover_dataset(), get_intensity_transform(), get_transforms(), GIDataset, main()

### Community 26 - "pandas"
Cohesion: 0.27
Nodes (8): build_table_1(), build_table_2(), build_table_3(), build_table_4(), fmt_md_mean_std(), fmt_mean_std(), main(), generate_audit_report()

### Community 27 - "FixLiarLabelAffined"
Cohesion: 0.13
Nodes (6): DiceCEWithIgnoreLoss, discover_phase2_dataset(), FixLiarLabelAffined, get_transforms(), main(), RemapGILabelTargets

### Community 28 - "_ResBlock3D"
Cohesion: 0.20
Nodes (3): _DownBlock3D, _ResBlock3D, _UpBlock3D

### Community 30 - "train_mednext_phase2.py"
Cohesion: 0.11
Nodes (6): DiceCEWithIgnoreLoss, discover_phase2_dataset(), FixLiarLabelAffined, get_transforms(), main(), RemapGILabelTargets

### Community 32 - "eval_scaling_checkpoints.py"
Cohesion: 0.22
Nodes (4): evaluate_checkpoint(), find_validation_cases(), main(), compute_dice_hd95()

### Community 33 - "numpy"
Cohesion: 0.05
Nodes (43): diagnose(), compute_betti_0(), compute_gpu_tensor_metrics(), compute_hd95_fast(), find_jhu_dir(), remap_prediction_labels(), run_gpu_evaluation(), evaluate_jhu_set() (+35 more)

### Community 34 - "run_clinical_turing_test.py"
Cohesion: 0.24
Nodes (4): compute_binary_dice(), main(), resample_real_to_target(), run_model_inference()

### Community 35 - "train_swin_unetr_phase2.py"
Cohesion: 0.16
Nodes (3): manual_mean_dice(), post_label_fn(), post_pred_fn()

### Community 36 - "load_stage1_autoencoder"
Cohesion: 0.24
Nodes (3): load_stage1_autoencoder(), restore_stage2_checkpoint(), stage2_checkpoint_payload()

### Community 38 - "predict_mednext.py"
Cohesion: 0.47
Nodes (3): discover_ct_files(), get_preprocessing(), main()

### Community 39 - "sbatch_train_mednext.sh"
Cohesion: 0.33
Nodes (5): CUDA_DEVICE_ORDER, OMP_NUM_THREADS, PYTHONPATH, PYTORCH_CUDA_ALLOC_CONF, sbatch_train_mednext.sh script

### Community 40 - "train_full_anchor_4gpu.sh"
Cohesion: 0.33
Nodes (5): MASTER_ADDR, MASTER_PORT, NCCL_TIMEOUT, PYTHONUNBUFFERED, train_full_anchor_4gpu.sh script

### Community 42 - "2. Detailed Per-Scan Clinical Annotations"
Cohesion: 0.18
Nodes (10): 1. Multi-Phase Mask Propagation Matrix, 2. Detailed Per-Scan Clinical Annotations, 3. Recommended Action Plan & Implementation Steps, `BDMAP_00242131` (Source Image: `note1.png`), `BDMAP_00242133` (Source Image: `note2.png`), `BDMAP_00242134` (Source Image: `note2.png`), `BDMAP_00242136` (Source Image: `note2.png`), `BDMAP_00394224` (Source Image: `notes3_for224.png`) (+2 more)

### Community 56 - "generate_multiphase_tumors.py"
Cohesion: 0.27
Nodes (3): extract_host_parenchyma_mask(), find_organ_mask(), load_nifti_volume()

### Community 58 - "main"
Cohesion: 0.25
Nodes (5): checkpoint_sha256(), main(), mask_conditioning_identity(), parse_args(), setup_ddp()

### Community 59 - "predict_swin_unetr.py"
Cohesion: 0.47
Nodes (3): discover_ct_files(), get_preprocessing(), main()

### Community 64 - "Permission & Directive: Proactive Specialized Subagents"
Cohesion: 0.50
Nodes (3): Permission & Directive: Proactive Specialized Subagents, Registered Specialized Subagents:, Subagent Delegation & Proactivity Rule

## Knowledge Gaps
- **16 isolated node(s):** `PYTHONUNBUFFERED`, `PYTHONPATH`, `CUDA_DEVICE_ORDER`, `PYTORCH_CUDA_ALLOC_CONF`, `OMP_NUM_THREADS` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 423 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AsymmetricPDCELoss` connect `torch` to `train_swin_unetr_phase2.py`, `run_scaling_sweep.py`, `2. Detailed Per-Scan Clinical Annotations`, `train_mednext.py`, `FixLiarLabelAffined`, `train_mednext_phase2.py`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **What connects `PYTHONUNBUFFERED`, `PYTHONPATH`, `CUDA_DEVICE_ORDER` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `multiphase_tumor_kinetics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12857142857142856 - nodes in this community are weakly interconnected._
- **Why does `🛠️ Key Scripts & Execution Pipelines` connect `test_active_learning.py` to `unified_gastro_dataset.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `evaluate_multi_model_ensemble.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06382978723404255 - nodes in this community are weakly interconnected._
- **Why does `3. Multi-Model Ensemble & Automated Data Cleansing` connect `unified_gastro_dataset.py` to `evaluate_multi_model_ensemble.py`, `test_active_learning.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `generate_showcase.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0707070707070707 - nodes in this community are weakly interconnected._