# Multi-Model Ensemble Fix & Automated Cleansing Plan

## Goal Description
Based on NotebookLM's analysis, we are officially pivoting from a human-in-the-loop Active Learning queue to a **Fully Automated Data Cleansing & Weakly-Supervised Distillation** pipeline. Our architecture is designed to be "under-annotation-tolerant," meaning we can leverage our massive data volume (22,000+ scans) to simply filter out noisy data mathematically rather than manually correcting it. 

This plan details how we will restructure the ensemble evaluation script to act as a rigorous data-cleansing engine, and outline the subsequent training steps using Teacher-Student Distillation and our asymmetric loss function.

## User Review Required
> [!IMPORTANT]
> **Radiologist Queue Eliminated**: This updated plan removes the active learning radiologist queue completely. Hard, uncertain edge cases will now be explicitly dropped (`NOISE_REJECT`), and coarse/weak annotations will utilize an `ignore` class for conflicting pixels. 
> 
> Please review this new fully-automated workflow. If this aligns perfectly with your "trying to auto exclude instead of correct" strategy, we can implement these changes in `evaluate_multi_model_ensemble.py` immediately.

## Proposed Strategy

### 1. The Fully Automated Pipeline
We will execute the following four steps to bypass human annotation and generate a rock-solid MedNeXt-B model:

1. **Auto-Exclude the Hard Cases**: Cases hitting our uncertainty / topological discrepancy thresholds ($|\Delta\beta_0| > 5$) will be explicitly discarded to prevent poisoning the training pool.
2. **Teacher-Student Distillation (GKD)**: Using the tens of thousands of auto-approved TotalSegmentator masks, we will train MedNeXt-B via Generalizable Knowledge Distillation (Phase 2), bypassing human boundaries.
3. **Hard Thresholding Auto-Labeling**: For the 1,735 scans with coarse human labels, conflicting pixels (where human labels disagree with TotalSegmentator) will be assigned an **'ignore' class**. The loss function will ignore these pixels during training.
4. **Asymmetric Loss Integration**: The MedNeXt-B model will be trained exclusively with our **Asymmetric Partial-CE/Dice Loss**, which has empirically proven to boost recall by +10.7 points on sparse/coarse datasets.

### 2. `evaluate_multi_model_ensemble.py` (Script Updates)
We will rewrite the Triage Logic in the evaluation script to reflect this automated data cleansing strategy. The script will continue utilizing the robust 4-Model Multi-Architecture Ensemble (`TotalSegmentator`, `deepGI`, `Swin-UNETR`, `MedNeXt-B`).

#### [MODIFY] `02_Projects/mp-factory/code/evaluation/evaluate_multi_model_ensemble.py`
**Update Triage Logic and Actions (IMPLEMENTED):**

The triage rules are now centralized in a pure, unit-tested `triage_case()` helper
plus the `NOISE_*` / `CLEAN_*` module constants (single source of truth reused by
`merge_ensemble_chunks.py`). The strict binning is:

```python
NOISE_BETTI_DIFF_MAX  = 5     # |Δβ0| strictly greater than this -> NOISE_REJECT
NOISE_DICE_MIN        = 0.50  # mean consensus Dice strictly below this -> NOISE_REJECT
NOISE_UNCERTAINTY_MAX = 0.15  # mean predictive entropy above this -> NOISE_REJECT
CLEAN_DICE_MIN        = 0.82
CLEAN_INTER_MODEL_MIN = 0.85
CLEAN_BETTI_DIFF_MAX  = 2

def triage_case(mean_consensus_dice, mean_inter_model_dice, max_betti_diff, mean_uncertainty):
    if (max_betti_diff > NOISE_BETTI_DIFF_MAX
            or mean_consensus_dice < NOISE_DICE_MIN
            or mean_uncertainty > NOISE_UNCERTAINTY_MAX):
        category = 'NOISE_REJECT'          # Auto-Exclude (Discard from Training Pool)
    elif (mean_consensus_dice >= CLEAN_DICE_MIN
            and mean_inter_model_dice >= CLEAN_INTER_MODEL_MIN
            and max_betti_diff <= CLEAN_BETTI_DIFF_MAX):
        category = 'CLEAN_HIGH_CONFIDENCE' # Auto-Approve for GKD Distillation & VAE
    else:
        category = 'WEAK_COARSE'           # Hard-Threshold conflicting pixels to Ignore Class
    return category, TRIAGE_ACTIONS[category]
```

> Note: the hard-rejection gate uses `mean Dice < 0.50` (not `< 0.82`) so that
> coarse-but-usable human labels flow into `WEAK_COARSE` for ignore-class
> auto-labeling instead of being discarded. Only topological blow-ups
> (`|Δβ0| > 5`), true low-overlap failures (`Dice < 0.50`), or high predictive
> entropy (`> 0.15`) are dumped.

**Dataset splits (IMPLEMENTED):** `export_dataset_splits()` materializes, next to
`ensemble_audit_summary.csv` (or under `--splits_dir`):
`ensemble_split_{clean_high_confidence,weak_coarse,noise_reject}.txt/.csv`,
a combined `ensemble_split_train_pool.txt` (CLEAN+WEAK, REJECT excluded), and a
`dataset_splits.json` manifest recording the active thresholds and per-bucket counts.

---

## Verification Plan

### Automated Tests
* We will verify the script executes without syntax errors.

### Manual Verification
1. We will execute the updated `evaluate_multi_model_ensemble.py`.
2. Review `ensemble_audit_summary.csv` to ensure:
   - Scans with topological errors ($|\Delta\beta_0| > 5$) are strictly dumped into `NOISE_REJECT`.
   - The `CLEAN_HIGH_CONFIDENCE` bucket correctly populates the pristine subset for Phase 2 GKD Distillation.
   - The `WEAK_COARSE` bucket successfully identifies candidates for Hard Thresholding auto-labeling.
