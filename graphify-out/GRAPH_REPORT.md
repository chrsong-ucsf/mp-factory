# Graph Report - .  (2026-08-04)

## Corpus Check
- 21 files · ~13,217 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 196 nodes · 239 edges · 22 communities (19 shown, 3 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.74)
- Token cost: 0 input · 72,256 output

## Community Hubs (Navigation)
- Active Learning Triage Loop
- Ensemble Evaluation Metrics
- Ensemble Fix & Distillation Plan
- TotalSeg Pipeline Scripts
- Datasets & Repo Overview
- MedNeXt Training Data Pipeline
- Swin-UNETR Training Data Pipeline
- Phase-1 GI Audit Metrics
- GI Mask Quality Evaluation
- Model Comparison (MedNeXt vs Swin-UNETR)
- Swin-UNETR Batch Inference
- MedNeXt Batch Inference
- Pipeline Progress Monitor
- Ensemble CSV Chunk Merging
- Pipeline Output Verification
- Legacy Audit Submission Script

## God Nodes (most connected - your core abstractions)
1. `mp-factory README` - 16 edges
2. `evaluate_subject()` - 14 edges
3. `ActiveLearningPool` - 10 edges
4. `UncertaintyEstimator` - 9 edges
5. `ActiveLearningOrchestrator` - 7 edges
6. `GIDataset` - 7 edges
7. `GIDataset` - 7 edges
8. `Fully Automated Data Cleansing & Weakly-Supervised Distillation Pipeline` - 7 edges
9. `TotalSegmentator` - 6 edges
10. `evaluate_case_pair()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `submit_audit.sh` --semantically_similar_to--> `Fully Automated Data Cleansing & Weakly-Supervised Distillation Pipeline`  [INFERRED] [semantically similar]
  README.md → ENSEMBLE_FIX_PLAN.md
- `evaluate_gi_masks.py` --semantically_similar_to--> `Betti-0 Topological Discrepancy Threshold (|Δβ0| > 5)`  [INFERRED] [semantically similar]
  README.md → ENSEMBLE_FIX_PLAN.md
- `run_evaluation.sbatch` --references--> `evaluate_multi_model_ensemble.py`  [INFERRED]
  README.md → ENSEMBLE_FIX_PLAN.md
- `test_uncertainty_estimator()` --calls--> `UncertaintyEstimator`  [INFERRED]
  code/test_active_learning.py → code/active_learning/loop.py
- `test_active_learning_pool()` --calls--> `ActiveLearningPool`  [INFERRED]
  code/test_active_learning.py → code/active_learning/loop.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **4-Model Multi-Architecture Ensemble Composition** — 02_projects_mp_factory_ensemble_fix_plan_four_model_ensemble, 02_projects_mp_factory_ensemble_fix_plan_totalsegmentator, 02_projects_mp_factory_ensemble_fix_plan_deepgi, 02_projects_mp_factory_ensemble_fix_plan_swin_unetr, 02_projects_mp_factory_ensemble_fix_plan_mednext_b [EXTRACTED 1.00]
- **Fully Automated Pipeline Four-Step Flow** — 02_projects_mp_factory_ensemble_fix_plan_automated_data_cleansing_pipeline, 02_projects_mp_factory_ensemble_fix_plan_noise_reject, 02_projects_mp_factory_ensemble_fix_plan_gkd_teacher_student_distillation, 02_projects_mp_factory_ensemble_fix_plan_hard_thresholding_auto_labeling, 02_projects_mp_factory_ensemble_fix_plan_asymmetric_partial_ce_dice_loss [EXTRACTED 1.00]
- **Triage Logic Categorization Scheme** — 02_projects_mp_factory_code_evaluate_multi_model_ensemble_triage_logic, 02_projects_mp_factory_ensemble_fix_plan_noise_reject, 02_projects_mp_factory_ensemble_fix_plan_clean_high_confidence, 02_projects_mp_factory_ensemble_fix_plan_weak_coarse [EXTRACTED 1.00]
- **Automated 3D Segmentation Workflows** — agents_run_totalsegmentator_batch_py, agents_run_totalseg_py, agents_run_totalseg_bdmap_py, agents_run_totalseg_gi_array_py, agents_totalsegmentator [EXTRACTED 1.00]

## Communities (22 total, 3 thin omitted)

### Community 0 - "Active Learning Triage Loop"
Cohesion: 0.08
Nodes (20): active_learning package  Exposes UncertaintyEstimator, ActiveLearningPool, and A, ActiveLearningOrchestrator, ActiveLearningPool, active_learning/loop.py  Core Active Learning Module for Triage, Uncertainty Est, Executes one Active Learning iteration:           1. Parses model comparison / e, Estimates voxel-level and scan-level uncertainty metrics., Computes spatial predictive entropy H(p) = - sum_c p_c log2(p_c)         probs:, Computes Monte Carlo Dropout variance across multiple forward passes. (+12 more)

### Community 1 - "Ensemble Evaluation Metrics"
Cohesion: 0.09
Nodes (30): compute_betti_0(), compute_classification_metrics(), compute_clustering_metrics(), compute_composite_score(), compute_diversity_weights(), compute_ece(), compute_hd95_fast(), compute_spatial_uncertainty() (+22 more)

### Community 2 - "Ensemble Fix & Distillation Plan"
Cohesion: 0.12
Nodes (21): evaluate_multi_model_ensemble.py, Triage / Categorization Logic, run_evaluation.sbatch, ensemble_audit_summary.csv, Multi-Model Ensemble Fix & Automated Cleansing Plan, Human-in-the-Loop Active Learning Radiologist Queue, Asymmetric Partial-CE/Dice Loss, Fully Automated Data Cleansing & Weakly-Supervised Distillation Pipeline (+13 more)

### Community 3 - "TotalSeg Pipeline Scripts"
Cohesion: 0.17
Nodes (18): download.sbatch, evaluate_gi_masks.py, run_totalseg.sbatch, run_totalseg_bdmap.sbatch, run_totalseg_gi_array.sbatch, smoke_test.sbatch, submit_audit.sh, main.py (SyntheticTumors trainer) (+10 more)

### Community 4 - "Datasets & Repo Overview"
Cohesion: 0.18
Nodes (11): BodyMaps, CancerVerse, evaluate_gi_masks.py, mp-factory, run_totalseg_bdmap.py, run_totalseg_gi_array.py, run_totalseg.py, run_totalsegmentator_batch.py (+3 more)

### Community 5 - "MedNeXt Training Data Pipeline"
Cohesion: 0.24
Nodes (7): discover_dataset(), get_transforms(), GIDataset, main(), Dataset, Find scans and assemble image/label pairs from CancerVerse subfolder structure s, Custom Dataset handler to merge separate organ NIfTI files on-the-fly if needed.

### Community 6 - "Swin-UNETR Training Data Pipeline"
Cohesion: 0.24
Nodes (7): discover_dataset(), get_transforms(), GIDataset, main(), Dataset, Find scans and assemble image/label pairs from CancerVerse subfolder structure s, Custom Dataset handler to merge separate organ NIfTI files on-the-fly if needed.

### Community 7 - "Phase-1 GI Audit Metrics"
Cohesion: 0.31
Nodes (9): compute_betti_0(), compute_clustering_metrics(), compute_hd95_fast(), discover_pairs(), evaluate_case_pair(), main(), Compute 0th Betti number (number of 3D connected components)., Fast EDT-based 95th Percentile Hausdorff Distance with proper morphological surf (+1 more)

### Community 8 - "GI Mask Quality Evaluation"
Cohesion: 0.39
Nodes (7): compute_betti_0(), compute_clustering_metrics(), compute_hd95(), evaluate_single_subject(), main(), Compute Betti-0 (number of connected components)., Compute 95th percentile Hausdorff Distance (HD95).

### Community 9 - "Model Comparison (MedNeXt vs Swin-UNETR)"
Cohesion: 0.38
Nodes (6): dice_score(), load_ground_truth(), main(), compare_models.py  Compares MedNeXt vs Swin-UNETR GI segmentation predictions ag, Compute binary Dice score for a single organ class., Merge individual organ segmentation NIfTIs into a single label map.

### Community 10 - "Swin-UNETR Batch Inference"
Cohesion: 0.47
Nodes (5): discover_ct_files(), get_preprocessing(), main(), predict_swin_unetr.py  Runs batch inference with a trained Swin-UNETR checkpoint, Find all ct.nii.gz files and shard them across parallel workers.

### Community 11 - "MedNeXt Batch Inference"
Cohesion: 0.60
Nodes (4): discover_ct_files(), get_preprocessing(), main(), predict_mednext.py  Batch inference with a trained MedNeXt checkpoint across all

### Community 12 - "Pipeline Progress Monitor"
Cohesion: 0.67
Nodes (3): get_progress(), main(), watch_pipeline.py  Monitors active pipeline jobs (Model Comparison & Active Lear

## Ambiguous Edges - Review These
- `evaluate_gi_masks.py` → `submit_audit.sh`  [AMBIGUOUS]
  README.md · relation: references

## Knowledge Gaps
- **19 isolated node(s):** `submit_audit.sh script`, `CancerVerse`, `BodyMaps`, `SyntheticTumors`, `submit_audit.sh` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `evaluate_gi_masks.py` and `submit_audit.sh`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `mp-factory README` connect `TotalSeg Pipeline Scripts` to `Ensemble Fix & Distillation Plan`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `Fully Automated Data Cleansing & Weakly-Supervised Distillation Pipeline` connect `Ensemble Fix & Distillation Plan` to `TotalSeg Pipeline Scripts`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `ActiveLearningPool` (e.g. with `main()` and `test_active_learning_orchestrator()`) actually correct?**
  _`ActiveLearningPool` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `UncertaintyEstimator` (e.g. with `main()` and `test_active_learning_orchestrator()`) actually correct?**
  _`UncertaintyEstimator` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ActiveLearningOrchestrator` (e.g. with `main()` and `test_active_learning_orchestrator()`) actually correct?**
  _`ActiveLearningOrchestrator` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `submit_audit.sh script`, `CancerVerse`, `BodyMaps` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._