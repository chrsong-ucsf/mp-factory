# Subagent Delegation & Proactivity Rule

## Permission & Directive: Proactive Specialized Subagents

The agent has full, proactive permission to define and spin up specialized subagents (`define_subagent`, `invoke_subagent`) whenever a repetitive, intensive, multi-step, or heavy background task is needed.

### Registered Specialized Subagents:

1. **`slurm_job_sentinel`** (Cluster Queue & Failure Auditor)
   - **Role**: Periodically monitors SLURM jobs (`squeue`, `sacct`), tails active `logs/*.out` and `logs/*.err`, detects silent worker halts, cgroup OOM kills, or CUDA exceptions early, and reports job completion or failure signatures immediately.
   - **Trigger**: Whenever multi-hour SLURM batch jobs or arrays are dispatched.

2. **`nifti_integrity_auditor`** (Volumetric Data QA)
   - **Role**: Runs header checks, shape compatibility, spacing/affine alignments, label validity (0..4, ignore_index=255), and NaN/Inf detection across CT volumes and multi-organ masks prior to training or distillation passes (`unified_gastro_dataset.py --verify`).
   - **Trigger**: Before dispatching new model training arrays or after bulk autolabeling passes.

3. **`phase2_training_watcher`** (Convergence & Metric Tracker)
   - **Role**: Inspects MedNeXt-B / Swin-UNETR validation loss curves, per-organ Dice trends (stomach, duodenum, small bowel, colon), and checkpoint generation across all 4 folds, alerting if validation plateauing or gradient collapse occurs.
   - **Trigger**: During active Phase 2 GPU distillation training passes.

4. **`jhu_radiologist_benchmark_auditor`** (Expert Validation Specialist)
   - **Role**: Evaluates held-out expert test predictions against `.seg.nrrd` radiologist ground-truth, applies multi-phase propagation rules (`evaluate_jhu_radiologist_gpu.py`), and compiles markdown/LaTeX comparison tables (Dice, HD95, Betti-0).
   - **Trigger**: Once model checkpoints are generated and ready for clinical validation.

5. **`synthetic_tumor_generator`** (Project 10 Pipeline Handler)
   - **Role**: Dedicated agent for SyntheticTumors / Project 10 tasks—handling SITK multiphase CT registrations, procedural parenchyma masking, pharmacokinetic contrast kinetics, and 3D autoencoder training without polluting the GI segmentation context.
   - **Trigger**: When running tumor simulation, registration, or Project 10 model training.

6. **`graphify_updater`** (Knowledge Graph Architect)
   - **Role**: Maintains and synchronizes the code knowledge graph in `graphify-out/` (`graph.json`, `graph.html`, `GRAPH_REPORT.md`) after structural code changes.
   - **Trigger**: After major architectural refactors, module additions, or directory reorganizations.

7. **`progress_reporter`** (Pipeline Milestone & Status Summarizer)
   - **Role**: Lightweight progress reporting subagent that audits active SLURM queues, inspects recent job completions, parses convergence metrics/logs, checks checkpoint generation on disk, and compiles a concise markdown progress report for the user.
   - **Trigger**: On-demand progress tracking, pipeline milestone checks, or periodic executive reporting.
