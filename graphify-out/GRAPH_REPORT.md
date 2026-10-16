# Graph Report - mp-factory  (2026-10-13)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 735 nodes · 1591 edges · 56 communities (38 shown, 18 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `46aa4145`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- multiphase_tumor_kinetics.py
- evaluate_multi_model_ensemble.py
- pandas
- generate_showcase.py
- run_scaling_sweep.py
- mp-factory README
- ActiveLearningPool
- torch
- register_multiphase_pairs.py
- os
- train_stage2_synthesis.py
- test_ensemble_evaluation.py
- numpy
- unified_gastro_dataset.py
- evaluate_synthesis.py
- train_stage1_autoencoder.py
- sample_multiphase_diffusion.py
- MedNeXt
- train_mednext_phase2.py
- train_swin_unetr.py
- evaluate_gi_masks.py
- audit_phase1_gi.py
- predict_swin_unetr.py
- autoencoder_3d.py
- Tensor
- train_mednext.py
- glob
- monitor_sweep_val.py
- _ResBlock3D
- run_gpu_evaluation
- FixLiarLabelAffined
- main
- eval_scaling_checkpoints.py
- compare_models.py
- run_step4_boundary_split.py
- test_val_loop_smoke.py
- evaluate_jhu_radiologist_set.py
- MultiphasePairedDataset
- predict_mednext.py
- sbatch_train_mednext.sh
- train_full_anchor_4gpu.sh
- run_p10_discovery.sh
- launch_qwen.sh
- eval_option_a.sh
- eval_option_b.sh
- submit_scaling_sweep.sh
- train_full_anchor.sh

## God Nodes (most connected - your core abstractions)
1. `mp-factory README` - 16 edges
2. `AsymmetricPDCELoss` - 15 edges
3. `evaluate_subject()` - 15 edges
4. `main()` - 13 edges
5. `process_subject()` - 13 edges
6. `ActiveLearningPool` - 10 edges
7. `main()` - 10 edges
8. `register_pair()` - 10 edges
9. `MultiPhaseTumorAugment` - 9 edges
10. `LightweightAutoEncoder3D` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_roundtrip_dataloader()` --uses--> `UnifiedGI3Dataset`  [INFERRED]
  code/utilities/test_unified_gastro_dataset.py → code/evaluation/unified_gastro_dataset.py
- `main()` --uses--> `AnatomyConditionalSynthesisNet`  [INFERRED]
  code/project10/evaluation/sample_multiphase_diffusion.py → code/project10/models/synthesis_net.py
- `main()` --uses--> `AnatomyConditionalSynthesisNet`  [INFERRED]
  code/project10/training/train_stage2_synthesis.py → code/project10/models/synthesis_net.py
- `main()` --uses--> `AutoEncoder3DConfig`  [INFERRED]
  code/project10/evaluation/sample_multiphase_diffusion.py → code/project10/models/autoencoder_3d.py
- `main()` --uses--> `AutoEncoder3DConfig`  [INFERRED]
  code/project10/training/train_stage1_autoencoder.py → code/project10/models/autoencoder_3d.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Automated 3D Segmentation Workflows** — code_prediction_run_totalsegmentator_batch, code_slurm_scripts_run_totalseg, code_slurm_scripts_run_totalseg_bdmap, code_slurm_scripts_run_totalseg_gi_array, agents_totalsegmentator [EXTRACTED 1.00]
- **Fully Automated Pipeline Four-Step Flow** — 02_projects_mp_factory_ensemble_fix_plan_automated_data_cleansing_pipeline, 02_projects_mp_factory_ensemble_fix_plan_noise_reject, 02_projects_mp_factory_ensemble_fix_plan_gkd_teacher_student_distillation, 02_projects_mp_factory_ensemble_fix_plan_hard_thresholding_auto_labeling, 02_projects_mp_factory_ensemble_fix_plan_asymmetric_partial_ce_dice_loss [EXTRACTED 1.00]
- **4-Model Multi-Architecture Ensemble Composition** — 02_projects_mp_factory_ensemble_fix_plan_four_model_ensemble, 02_projects_mp_factory_ensemble_fix_plan_totalsegmentator, 02_projects_mp_factory_ensemble_fix_plan_deepgi, 02_projects_mp_factory_ensemble_fix_plan_swin_unetr, 02_projects_mp_factory_ensemble_fix_plan_mednext_b [EXTRACTED 1.00]
- **Triage Logic Categorization Scheme** — 02_projects_mp_factory_code_evaluate_multi_model_ensemble_triage_logic, 02_projects_mp_factory_ensemble_fix_plan_noise_reject, 02_projects_mp_factory_ensemble_fix_plan_clean_high_confidence, 02_projects_mp_factory_ensemble_fix_plan_weak_coarse [EXTRACTED 1.00]

## Communities (56 total, 18 thin omitted)

### Community 0 - "multiphase_tumor_kinetics.py"
Cohesion: 0.06
Nodes (12): extract_host_parenchyma_mask(), find_organ_mask(), load_nifti_volume(), main(), _add_texture_noise(), apply_kinetics(), _get_ring_mask(), _make_ellipsoid_mask() (+4 more)

### Community 1 - "evaluate_multi_model_ensemble.py"
Cohesion: 0.05
Nodes (34): Triage / Categorization Logic, ensemble_audit_summary.csv, Multi-Model Ensemble Fix & Automated Cleansing Plan, Human-in-the-Loop Active Learning Radiologist Queue, Asymmetric Partial-CE/Dice Loss, Fully Automated Data Cleansing & Weakly-Supervised Distillation Pipeline, Betti-0 Topological Discrepancy Threshold (|Δβ0| > 5), CLEAN_HIGH_CONFIDENCE Triage Category (+26 more)

### Community 2 - "pandas"
Cohesion: 0.11
Nodes (12): fit_and_print(), power_law(), export_dataset_splits(), main(), main(), build_manifest(), normalise_date(), parse_args() (+4 more)

### Community 3 - "generate_showcase.py"
Cohesion: 0.11
Nodes (9): plot_curves(), apply_window(), generate_interactive_3d_html(), process_subject(), render_3d_multiangle(), render_orthoview_figure(), render_progression_montage(), render_radiologist_comparison() (+1 more)

### Community 4 - "run_scaling_sweep.py"
Cohesion: 0.11
Nodes (14): MedNeXtB(), build_model(), build_transforms(), compute_dice_hd95(), _compute_hd95_numpy(), create_dummy_cohort(), evaluate_radiologist_true_dice(), _extract_3d_boundaries() (+6 more)

### Community 5 - "mp-factory README"
Cohesion: 0.10
Nodes (22): download.sbatch, run_evaluation.sbatch, run_totalseg.sbatch, run_totalseg_bdmap.sbatch, run_totalseg_gi_array.sbatch, smoke_test.sbatch, main.py (SyntheticTumors trainer), mp-factory README (+14 more)

### Community 6 - "ActiveLearningPool"
Cohesion: 0.08
Nodes (5): ActiveLearningOrchestrator, ActiveLearningPool, UncertaintyEstimator, test_active_learning_pool(), test_uncertainty_estimator()

### Community 7 - "torch"
Cohesion: 0.14
Nodes (8): AsymmetricPDCELoss, test_asymmetric_penalty_fn_gt_fp(), test_gradient_flows_to_backbone(), __init__(), test_ignore_index_masks_voxels(), test_ignore_index_minus_one_noop(), test_multiclass_3d_shape(), test_probability_clamping_no_nan()

### Community 8 - "register_multiphase_pairs.py"
Cohesion: 0.15
Nodes (8): affine_registration(), bspline_registration(), load_sitk_image(), main(), parse_args(), register_pair(), resample_to_fixed(), rigid_registration()

### Community 9 - "os"
Cohesion: 0.16
Nodes (10): autolabel_subject(), main(), resolve_human_label(), _require_nibabel(), test_autolabel_e2e_synthetic(), test_ignore_index_consistency(), test_organ_classes(), test_resolve_human_label_combined() (+2 more)

### Community 10 - "train_stage2_synthesis.py"
Cohesion: 0.11
Nodes (4): DiffusionSchedule, main(), parse_args(), setup_ddp()

### Community 11 - "test_ensemble_evaluation.py"
Cohesion: 0.17
Nodes (10): test_active_learning_orchestrator(), _require_nibabel(), test_export_dataset_splits(), test_full_pipeline_mock(), _sample_audit(), test_build_manifest_clean(), test_build_manifest_with_weak(), test_manifest_ignore_index_match() (+2 more)

### Community 12 - "numpy"
Cohesion: 0.22
Nodes (7): diagnose(), compute_spatial_uncertainty(), compute_subject(), verify_predictions(), test_spatial_uncertainty(), fix_consensus(), fix_gi_mask()

### Community 13 - "unified_gastro_dataset.py"
Cohesion: 0.13
Nodes (5): build_manifest(), main(), UnifiedGI3Dataset, verify_dataset_integrity(), test_verify_report_nonexistent()

### Community 14 - "evaluate_synthesis.py"
Cohesion: 0.18
Nodes (6): compute_metrics_3d(), find_matching_real_file(), load_nifti(), main(), merge_worker_csvs(), resample_real_to_target()

### Community 15 - "train_stage1_autoencoder.py"
Cohesion: 0.15
Nodes (7): normalise_hu(), _center_crop_pad(), main(), MultiPhaseVolumeDataset, parse_args(), _resample_to_isotropic(), setup_ddp()

### Community 16 - "sample_multiphase_diffusion.py"
Cohesion: 0.15
Nodes (4): DDIMSampler, load_anatomy_mask(), main(), preprocess_ct()

### Community 17 - "MedNeXt"
Cohesion: 0.13
Nodes (4): GRN, LayerNorm, MedNeXt, MedNeXtBlock

### Community 19 - "train_swin_unetr.py"
Cohesion: 0.15
Nodes (5): discover_dataset(), get_transforms(), GIDataset, main(), RemapGILabelTargets

### Community 20 - "evaluate_gi_masks.py"
Cohesion: 0.18
Nodes (7): compute_betti_0(), compute_clustering_metrics(), compute_hd95(), compute_volumetric_dice(), evaluate_single_subject(), extract_3d_boundaries(), main()

### Community 21 - "audit_phase1_gi.py"
Cohesion: 0.20
Nodes (6): compute_betti_0(), compute_clustering_metrics(), compute_hd95_fast(), discover_pairs(), evaluate_case_pair(), main()

### Community 22 - "predict_swin_unetr.py"
Cohesion: 0.16
Nodes (4): discover_ct_files(), get_preprocessing(), main(), AnatomyConditionalSynthesisNet

### Community 23 - "autoencoder_3d.py"
Cohesion: 0.16
Nodes (4): AutoEncoder3DConfig, build_autoencoder(), denormalise_hu(), reconstruction_loss()

### Community 25 - "train_mednext.py"
Cohesion: 0.19
Nodes (5): discover_dataset(), get_intensity_transform(), get_transforms(), GIDataset, main()

### Community 26 - "glob"
Cohesion: 0.26
Nodes (4): find_ct_file(), main(), main(), main()

### Community 27 - "monitor_sweep_val.py"
Cohesion: 0.27
Nodes (5): get_progress(), main(), display_status(), main(), parse_job_log()

### Community 28 - "_ResBlock3D"
Cohesion: 0.20
Nodes (3): _DownBlock3D, _ResBlock3D, _UpBlock3D

### Community 29 - "run_gpu_evaluation"
Cohesion: 0.20
Nodes (5): compute_betti_0(), compute_gpu_tensor_metrics(), compute_hd95_fast(), find_jhu_dir(), run_gpu_evaluation()

### Community 30 - "FixLiarLabelAffined"
Cohesion: 0.22
Nodes (3): FixLiarLabelAffined, get_transforms(), RemapGILabelTargets

### Community 31 - "main"
Cohesion: 0.22
Nodes (3): DiceCEWithIgnoreLoss, discover_phase2_dataset(), main()

### Community 32 - "eval_scaling_checkpoints.py"
Cohesion: 0.32
Nodes (3): evaluate_checkpoint(), find_validation_cases(), main()

### Community 33 - "compare_models.py"
Cohesion: 0.36
Nodes (3): dice_score(), load_ground_truth(), main()

### Community 34 - "run_step4_boundary_split.py"
Cohesion: 0.32
Nodes (3): main(), process_subject(), _worker_process()

### Community 35 - "test_val_loop_smoke.py"
Cohesion: 0.25
Nodes (3): manual_mean_dice(), post_label_fn(), post_pred_fn()

### Community 36 - "evaluate_jhu_radiologist_set.py"
Cohesion: 0.48
Nodes (4): compute_betti_0(), dice_score(), evaluate_jhu_set(), iou_score()

### Community 38 - "predict_mednext.py"
Cohesion: 0.47
Nodes (3): discover_ct_files(), get_preprocessing(), main()

### Community 39 - "sbatch_train_mednext.sh"
Cohesion: 0.33
Nodes (5): CUDA_DEVICE_ORDER, OMP_NUM_THREADS, PYTHONPATH, PYTORCH_CUDA_ALLOC_CONF, sbatch_train_mednext.sh script

### Community 40 - "train_full_anchor_4gpu.sh"
Cohesion: 0.33
Nodes (5): MASTER_ADDR, MASTER_PORT, NCCL_TIMEOUT, PYTHONUNBUFFERED, train_full_anchor_4gpu.sh script

## Knowledge Gaps
- **22 isolated node(s):** `CUDA_DEVICE_ORDER`, `OMP_NUM_THREADS`, `PYTHONPATH`, `PYTORCH_CUDA_ALLOC_CONF`, `MASTER_ADDR` (+17 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 318 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `mp-factory README` connect `mp-factory README` to `evaluate_gi_masks.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **What connects `CUDA_DEVICE_ORDER`, `OMP_NUM_THREADS`, `PYTHONPATH` to the rest of the system?**
  _22 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `multiphase_tumor_kinetics.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06184012066365008 - nodes in this community are weakly interconnected._
- **Why does `main()` connect `multiphase_tumor_kinetics.py` to `pandas`, `numpy`, `torch`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Should `evaluate_multi_model_ensemble.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05176470588235294 - nodes in this community are weakly interconnected._
- **Why does `ActiveLearningPool` connect `ActiveLearningPool` to `pandas`, `test_ensemble_evaluation.py`, `torch`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Should `pandas` be split into smaller, more focused modules?**
  _Cohesion score 0.11260504201680673 - nodes in this community are weakly interconnected._