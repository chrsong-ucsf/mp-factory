# Graph Report - mp-factory  (2026-10-31)

## Corpus Check
- 103 files · ~1,788,196 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 77 file(s) not represented in the graph (top: .sbatch 67, .nrrd 6, (none) 2)

## Summary
- 988 nodes · 2123 edges · 73 communities (48 shown, 25 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 82 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8025a4fa`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- multiphase_tumor_kinetics.py
- evaluate_multi_model_ensemble.py
- json
- generate_showcase.py
- run_scaling_sweep.py
- evaluate_gi_masks.py
- pathlib
- torch
- register_multiphase_pairs.py
- os
- train_stage2_synthesis.py
- test_unified_gastro_dataset.py
- PART III: Master Publication Tables (Tables 1–4)
- unified_gastro_dataset.py
- evaluate_synthesis.py
- MultiPhaseVolumeDataset
- sample_multiphase_diffusion.py
- torch_nn_functional
- FixLiarLabelAffined
- GIDataset
- test_stage2_checkpoint.py
- audit_phase1_gi.py
- AnatomyConditionalSynthesisNet
- stress_test_contrast_perturbation.py
- Tensor
- train_mednext.py
- pandas
- train_swin_unetr_phase2.py
- _ResBlock3D
- TestPipelineDryRun
- main
- numpy
- train_mednext_phase2.py
- test_multiphase_paired_dataset.py
- run_model_inference
- test_val_loop_smoke.py
- main
- MultiphasePairedDataset
- run_clinical_turing_test.py
- sbatch_train_mednext.sh
- train_full_anchor_4gpu.sh
- run_p10_discovery.sh
- 2. Detailed Per-Scan Clinical Annotations
- eval_option_a.sh
- eval_option_b.sh
- submit_scaling_sweep.sh
- train_full_anchor.sh
- evaluate_jhu_radiologist_gpu.py
- sys
- glob
- predict_swin_unetr.py
- test_totalseg_label_maps.py
- DiffusionSchedule
- test_ensemble_evaluation.py
- MedNeXt
- Permission & Directive: Proactive Specialized Subagents
- compute_scaling_law.py
- csv
- BDMAP Radiologist Mask Reuse Notes
- mp-factory

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

## Communities (73 total, 25 thin omitted)

### Community 0 - "multiphase_tumor_kinetics.py"
Cohesion: 0.06
Nodes (12): extract_host_parenchyma_mask(), find_organ_mask(), load_nifti_volume(), main(), _add_texture_noise(), apply_kinetics(), _get_ring_mask(), _make_ellipsoid_mask() (+4 more)

### Community 1 - "evaluate_multi_model_ensemble.py"
Cohesion: 0.11
Nodes (13): adjusted_rand_score_fast(), compute_betti_0(), compute_classification_metrics(), compute_clustering_metrics(), compute_composite_score(), compute_ece(), compute_hd95_fast(), compute_surface_metrics() (+5 more)

### Community 2 - "json"
Cohesion: 0.17
Nodes (5): compute_subject(), main(), build_manifest(), normalise_date(), parse_args()

### Community 3 - "generate_showcase.py"
Cohesion: 0.07
Nodes (14): plot_curves(), apply_ct_window(), generate_comparison_figure(), generate_tumor_showcase_figure(), main(), apply_window(), generate_interactive_3d_html(), main() (+6 more)

### Community 4 - "run_scaling_sweep.py"
Cohesion: 0.12
Nodes (12): build_transforms(), compute_dice_hd95(), _compute_hd95_numpy(), create_dummy_cohort(), evaluate_radiologist_true_dice(), _extract_3d_boundaries(), find_cases(), load_gold_standard_nrrd_cases() (+4 more)

### Community 5 - "evaluate_gi_masks.py"
Cohesion: 0.09
Nodes (22): download.sbatch, run_evaluation.sbatch, run_totalseg.sbatch, run_totalseg_bdmap.sbatch, run_totalseg_gi_array.sbatch, smoke_test.sbatch, main.py (SyntheticTumors trainer), mp-factory README (+14 more)

### Community 6 - "pathlib"
Cohesion: 0.05
Nodes (22): 1. Data Ingestion & Management, 2. Automated 3D Segmentation, 4. Model Training (Phase 1 & Phase 2 Distillation), 5. Comparison & Expert Validation, 6. Monitoring & Active Learning, 7. Synthetic Tumor Submodule (`code/SyntheticTumors/`), AGENTS.md — Workspace Architecture & Project Guide, 📂 Codebase Directory Layout (+14 more)

### Community 7 - "torch"
Cohesion: 0.22
Nodes (9): AsymmetricPDCELoss, compute_dice_hd95(), test_asymmetric_penalty_fn_gt_fp(), test_gradient_flows_to_backbone(), __init__(), test_ignore_index_masks_voxels(), test_ignore_index_minus_one_noop(), test_multiclass_3d_shape() (+1 more)

### Community 8 - "register_multiphase_pairs.py"
Cohesion: 0.12
Nodes (10): get_progress(), main(), affine_registration(), bspline_registration(), load_sitk_image(), main(), parse_args(), register_pair() (+2 more)

### Community 9 - "os"
Cohesion: 0.14
Nodes (10): autolabel_subject(), main(), resolve_human_label(), _require_nibabel(), test_autolabel_e2e_synthetic(), test_ignore_index_consistency(), test_organ_classes(), test_resolve_human_label_combined() (+2 more)

### Community 10 - "train_stage2_synthesis.py"
Cohesion: 0.17
Nodes (4): reconstruction_loss(), main(), parse_args(), setup_ddp()

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
Cohesion: 0.09
Nodes (12): compute_metrics_3d(), find_matching_real_file(), load_nifti(), main(), merge_worker_csvs(), resample_real_to_target(), test_compile_manuscript_tables_patient_counting(), test_jhu_artifact_filtering() (+4 more)

### Community 15 - "MultiPhaseVolumeDataset"
Cohesion: 0.18
Nodes (4): normalise_hu(), _center_crop_pad(), MultiPhaseVolumeDataset, _resample_to_isotropic()

### Community 16 - "sample_multiphase_diffusion.py"
Cohesion: 0.08
Nodes (8): get_common_spatial_transform(), preprocess_single_volume(), checkpoint_sha256(), DDIMSampler, load_anatomy_mask(), main(), preprocess_ct(), denormalise_hu()

### Community 17 - "torch_nn_functional"
Cohesion: 0.18
Nodes (3): GRN, LayerNorm, MedNeXtBlock

### Community 19 - "GIDataset"
Cohesion: 0.13
Nodes (5): discover_dataset(), get_transforms(), GIDataset, main(), RemapGILabelTargets

### Community 20 - "test_stage2_checkpoint.py"
Cohesion: 0.12
Nodes (15): test_legacy_resume_continues_epoch_and_lr_without_new_fields(), test_legacy_resume_preserves_best_checkpoint_metric(), test_legacy_resume_requires_explicit_override(), test_main_rejects_missing_ae_before_empty_dataset(), test_new_checkpoint_records_resume_state_and_ae_identity(), test_resume_rejects_different_autoencoder_even_with_override(), test_resume_rejects_different_mask_conditioning_even_with_override(), test_resume_restores_scheduler_optimizer_and_best_loss() (+7 more)

### Community 21 - "audit_phase1_gi.py"
Cohesion: 0.11
Nodes (12): audit_section(), check_nifti_sample(), main(), compute_betti_0(), compute_clustering_metrics(), compute_hd95_fast(), discover_pairs(), evaluate_case_pair() (+4 more)

### Community 23 - "stress_test_contrast_perturbation.py"
Cohesion: 0.18
Nodes (6): apply_perturbation(), compute_binary_dice(), generate_markdown_report(), main(), plot_robustness_curves(), run_model_inference()

### Community 24 - "Tensor"
Cohesion: 0.15
Nodes (4): AutoEncoder3DConfig, build_autoencoder(), LightweightAutoEncoder3D, MONAI_VQVAE3D

### Community 25 - "train_mednext.py"
Cohesion: 0.19
Nodes (5): discover_dataset(), get_intensity_transform(), get_transforms(), GIDataset, main()

### Community 26 - "pandas"
Cohesion: 0.17
Nodes (11): main(), diagnose_failure_cause(), main(), build_table_1(), build_table_2(), build_table_3(), build_table_4(), fmt_md_mean_std() (+3 more)

### Community 27 - "train_swin_unetr_phase2.py"
Cohesion: 0.15
Nodes (6): DiceCEWithIgnoreLoss, discover_phase2_dataset(), FixLiarLabelAffined, get_transforms(), main(), RemapGILabelTargets

### Community 28 - "_ResBlock3D"
Cohesion: 0.20
Nodes (3): _DownBlock3D, _ResBlock3D, _UpBlock3D

### Community 30 - "main"
Cohesion: 0.11
Nodes (6): DiceCEWithIgnoreLoss, discover_phase2_dataset(), FixLiarLabelAffined, get_transforms(), main(), RemapGILabelTargets

### Community 31 - "numpy"
Cohesion: 0.15
Nodes (11): compute_diversity_weights(), compute_spatial_uncertainty(), bootstrap_ci(), cohens_d(), evaluate_pairs(), main(), inspect_label(), fix_consensus() (+3 more)

### Community 32 - "train_mednext_phase2.py"
Cohesion: 0.13
Nodes (3): evaluate_checkpoint(), find_validation_cases(), main()

### Community 33 - "test_multiphase_paired_dataset.py"
Cohesion: 0.30
Nodes (14): _dataset(), test_corrupt_mask_does_not_return_zero_sample(), test_full_body_mask_requires_version_provenance(), test_full_body_totalseg_ids_map_to_gi_labels(), test_low_full_body_ids_do_not_pass_as_gi_labels(), test_malformed_ok_row_raises(), test_missing_ct_raises(), test_missing_or_unusable_mask_raises() (+6 more)

### Community 34 - "run_model_inference"
Cohesion: 0.24
Nodes (4): compute_binary_dice(), main(), resample_real_to_target(), run_model_inference()

### Community 35 - "test_val_loop_smoke.py"
Cohesion: 0.25
Nodes (3): manual_mean_dice(), post_label_fn(), post_pred_fn()

### Community 36 - "main"
Cohesion: 0.14
Nodes (8): checkpoint_sha256(), load_stage1_autoencoder(), main(), mask_conditioning_identity(), parse_args(), restore_stage2_checkpoint(), setup_ddp(), stage2_checkpoint_payload()

### Community 38 - "run_clinical_turing_test.py"
Cohesion: 0.32
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

### Community 56 - "evaluate_jhu_radiologist_gpu.py"
Cohesion: 0.15
Nodes (7): compute_betti_0(), compute_gpu_tensor_metrics(), compute_hd95_fast(), find_ct_file(), find_jhu_dir(), run_gpu_evaluation(), evaluate_jhu_set()

### Community 57 - "sys"
Cohesion: 0.21
Nodes (5): main(), main(), main(), process_subject(), _worker_process()

### Community 58 - "glob"
Cohesion: 0.27
Nodes (6): TotalSegmentator, diagnose(), main(), verify_predictions(), main(), main()

### Community 59 - "predict_swin_unetr.py"
Cohesion: 0.38
Nodes (3): discover_ct_files(), get_preprocessing(), main()

### Community 60 - "test_totalseg_label_maps.py"
Cohesion: 0.18
Nodes (9): remap_prediction_labels(), normalize_prediction_mask(), test_ensemble_does_not_infer_schema_from_low_full_body_values(), test_ensemble_preserves_native_gi_mask(), test_ensemble_recognizes_totalseg_name_case_insensitively(), test_ensemble_rejects_unknown_full_body_schema(), test_ensemble_remaps_default_totalseg_and_ignores_non_gi_ids(), test_jhu_preserves_existing_gi_model_remap() (+1 more)

### Community 62 - "test_ensemble_evaluation.py"
Cohesion: 0.31
Nodes (6): triage_case(), _require_nibabel(), test_export_dataset_splits(), test_full_pipeline_mock(), test_triage_boundary_conditions(), test_triage_case_rules()

### Community 63 - "MedNeXt"
Cohesion: 0.29
Nodes (3): MedNeXt, MedNeXtB(), build_model()

### Community 64 - "Permission & Directive: Proactive Specialized Subagents"
Cohesion: 0.50
Nodes (3): Permission & Directive: Proactive Specialized Subagents, Registered Specialized Subagents:, Subagent Delegation & Proactivity Rule

### Community 66 - "csv"
Cohesion: 0.36
Nodes (3): dice_score(), load_ground_truth(), main()

### Community 69 - "mp-factory"
Cohesion: 0.50
Nodes (4): BodyMaps, CancerVerse, mp-factory, SyntheticTumors

## Knowledge Gaps
- **16 isolated node(s):** `PYTHONUNBUFFERED`, `PYTHONPATH`, `CUDA_DEVICE_ORDER`, `PYTORCH_CUDA_ALLOC_CONF`, `OMP_NUM_THREADS` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 438 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AsymmetricPDCELoss` connect `torch` to `train_mednext_phase2.py`, `run_scaling_sweep.py`, `.forward`, `2. Detailed Per-Scan Clinical Annotations`, `train_mednext.py`, `train_swin_unetr_phase2.py`, `main`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **What connects `PYTHONUNBUFFERED`, `PYTHONPATH`, `CUDA_DEVICE_ORDER` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `multiphase_tumor_kinetics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06108597285067873 - nodes in this community are weakly interconnected._
- **Why does `🛠️ Key Scripts & Execution Pipelines` connect `pathlib` to `unified_gastro_dataset.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `evaluate_multi_model_ensemble.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10582010582010581 - nodes in this community are weakly interconnected._
- **Why does `3. Multi-Model Ensemble & Automated Data Cleansing` connect `unified_gastro_dataset.py` to `evaluate_multi_model_ensemble.py`, `test_ensemble_evaluation.py`, `pathlib`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `generate_showcase.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06753006475485661 - nodes in this community are weakly interconnected._