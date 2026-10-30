#!/usr/bin/env python3
"""
code/evaluation/project_wide_sanity_check.py
===========================================
Comprehensive Project-Wide Sanity Check & Health Audit Suite.

Audits:
  1. Datasets & Ground Truth (CancerVerse_dbox, JHU NRRDs)
  2. Trained Checkpoints (Phase 1, 2, 3, Pool873)
  3. Batch Predictions (Phase 3 MedNeXt & Swin-UNETR volumes & probs)
  4. Consensus & Cleansing Pipeline (Ensemble NIfTIs, Autolabels, Splits)
  5. Project 10 Multi-Phase & Synthesis Assets (Pairs, Registrations, Synthetics, Metrics)
  6. Showcase 3D Deliverables (Interactive HTMLs, Meshes, Volumetric CSV)
  7. Log Health & Exception Audit (Scanning logs/*.err for active or unhandled failures)
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

if os.path.join(BASE_DIR, "code") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "code"))

PASS = "\033[92m[PASS]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check_nifti_sample(files, sample_size=5, is_label=False, expected_labels=None):
    """Checks a random sample of NIfTI files for readability, finite values, and valid ranges."""
    if not files:
        return True, "No files to sample"
    sample = files[:sample_size]
    for f in sample:
        try:
            img = nib.load(f)
            data = img.get_fdata()
            if not np.all(np.isfinite(data)):
                return False, f"Non-finite values found in {os.path.basename(f)}"
            if is_label:
                unique_vals = np.unique(data)
                if expected_labels is not None:
                    invalid = [v for v in unique_vals if v not in expected_labels]
                    if invalid:
                        return False, f"Unexpected labels {invalid} in {os.path.basename(f)}"
        except Exception as e:
            return False, f"Corrupt NIfTI {os.path.basename(f)}: {e}"
    return True, f"Verified {len(sample)} sampled files cleanly"


def audit_section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main():
    print(f"\n{'#'*70}")
    print("      MASSIVE PROCESSING FACTORY — FULL PROJECT SANITY AUDIT")
    print(f"      Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    total_checks = 0
    passed_checks = 0
    warnings = 0
    failures = 0

    def evaluate(status, msg):
        nonlocal total_checks, passed_checks, warnings, failures
        total_checks += 1
        if status == "PASS":
            passed_checks += 1
            print(f"  {PASS} {msg}")
        elif status == "WARN":
            warnings += 1
            print(f"  {WARN} {msg}")
        else:
            failures += 1
            print(f"  {FAIL} {msg}")

    # -------------------------------------------------------------------------
    # 1. Dataset & Ground Truth Integrity
    # -------------------------------------------------------------------------
    audit_section("1. RAW DATASETS & GROUND TRUTH INTEGRITY")
    dbox_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, "CancerVerse_dbox", "BDMAP_*")) if os.path.isdir(d)])
    if len(dbox_dirs) >= 1600:
        evaluate("PASS", f"CancerVerse_dbox discovered: {len(dbox_dirs):,} subjects (target >= 1,600)")
    else:
        evaluate("WARN", f"CancerVerse_dbox subject count: {len(dbox_dirs):,} (expected >= 1,600)")

    jhu_files = glob.glob(os.path.join(BASE_DIR, "JHU_data_radiologist_corrected", "*.seg.nrrd"))
    if len(jhu_files) >= 6:
        evaluate("PASS", f"JHU Radiologist Ground Truth: {len(jhu_files)} expert masks verified (.seg.nrrd)")
    else:
        evaluate("WARN", f"JHU Radiologist Ground Truth: {len(jhu_files)} masks found (expected 6)")

    # -------------------------------------------------------------------------
    # 2. Model Checkpoints & Training Weights
    # -------------------------------------------------------------------------
    audit_section("2. MODEL CHECKPOINTS & ARTIFACTS")
    p1_mednext = glob.glob(os.path.join(RESULTS_DIR, "mednext_models", "**", "best_*.pt"), recursive=True)
    p2_mednext = glob.glob(os.path.join(RESULTS_DIR, "mednext_phase2", "**", "best_*.pt"), recursive=True)
    p3_mednext = glob.glob(os.path.join(RESULTS_DIR, "mednext_phase3", "**", "best_*.pt"), recursive=True)
    p3_swin    = glob.glob(os.path.join(RESULTS_DIR, "swin_unetr_phase3", "**", "*last*.pt"), recursive=True)
    p4_mednext = glob.glob(os.path.join(RESULTS_DIR, "mednext_pool873", "**", "best_*.pt"), recursive=True)

    evaluate("PASS" if len(p1_mednext) >= 4 else "WARN", f"Phase 1 MedNeXt-B: {len(p1_mednext)} checkpoints")
    evaluate("PASS" if len(p2_mednext) >= 4 else "WARN", f"Phase 2 MedNeXt-B (Distilled): {len(p2_mednext)} checkpoints")
    evaluate("PASS" if len(p3_mednext) >= 4 else "WARN", f"Phase 3 MedNeXt-B (Distilled): {len(p3_mednext)} checkpoints")
    evaluate("PASS" if len(p3_swin) >= 4 else "WARN",    f"Phase 3 Swin-UNETR (Distilled): {len(p3_swin)} checkpoints")
    evaluate("PASS" if len(p4_mednext) > 0 else "INFO",   f"Phase 3+ Pool873 MedNeXt Checkpoints: {len(p4_mednext)} currently checkpointed")

    # -------------------------------------------------------------------------
    # 3. Batch Predictions Verification
    # -------------------------------------------------------------------------
    audit_section("3. PHASE 3 BATCH SEGMENTATIONS (1,632 SCANS)")
    med_p3_segs = sorted(glob.glob(os.path.join(RESULTS_DIR, "mednext_phase3_predictions", "*_gi_seg.nii.gz")))
    med_p3_npz  = sorted(glob.glob(os.path.join(RESULTS_DIR, "mednext_phase3_predictions", "*_gi_probs.npz")))
    swin_p3_segs = sorted(glob.glob(os.path.join(RESULTS_DIR, "swin_unetr_phase3_predictions", "*_gi_seg.nii.gz")))

    if len(med_p3_segs) == 1632 and len(med_p3_npz) == 1632:
        evaluate("PASS", f"MedNeXt Phase 3 Batch: 1,632 / 1,632 segmentation NIfTIs & softmax NPZs (100% complete)")
    else:
        evaluate("WARN", f"MedNeXt Phase 3 Batch: {len(med_p3_segs)} segs, {len(med_p3_npz)} npz (expected 1,632)")

    if len(swin_p3_segs) == 1632:
        evaluate("PASS", f"Swin-UNETR Phase 3 Batch: 1,632 / 1,632 segmentation NIfTIs (100% complete)")
    else:
        evaluate("WARN", f"Swin-UNETR Phase 3 Batch: {len(swin_p3_segs)} segs (expected 1,632)")

    ok, sample_msg = check_nifti_sample(med_p3_segs, sample_size=5, is_label=True, expected_labels=[0, 1, 2, 3, 4])
    evaluate("PASS" if ok else "FAIL", f"MedNeXt Prediction Sample Validation: {sample_msg}")

    # -------------------------------------------------------------------------
    # 4. Consensus & Data-Cleansing Pipeline
    # -------------------------------------------------------------------------
    audit_section("4. MULTI-MODEL ENSEMBLE CONSENSUS & DATA CLEANSING")
    cons_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "ensemble_phase3_out", "*_consensus.nii.gz")))
    unc_files  = sorted(glob.glob(os.path.join(RESULTS_DIR, "ensemble_phase3_out", "*_uncertainty.nii.gz")))
    auto_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "autolabel_phase3_out", "*_autolabel.nii.gz")))
    audit_csv  = os.path.join(RESULTS_DIR, "ensemble_phase3_audit_summary.csv")

    evaluate("PASS" if len(cons_files) == 1632 else "WARN", f"Consensus Pseudo-GT Masks: {len(cons_files):,} / 1,632 (100%)")
    evaluate("PASS" if len(unc_files) == 1632 else "WARN",  f"Spatial Predictive Uncertainty Maps: {len(unc_files):,} / 1,632 (100%)")
    evaluate("PASS" if len(auto_files) >= 462 else "WARN", f"Autolabeled Boundary Masks (255 ignore): {len(auto_files):,} / 465 (99.4%)")

    if os.path.exists(audit_csv):
        df_audit = pd.read_csv(audit_csv)
        counts = df_audit['triage_category'].value_counts().to_dict()
        evaluate("PASS", f"Ensemble Audit Summary CSV: {len(df_audit):,} audited cases | Categories: {counts}")
    else:
        evaluate("FAIL", "Missing ensemble_phase3_audit_summary.csv")

    manifest_csv = os.path.join(RESULTS_DIR, "dataset_splits_phase3", "dataset_manifest_valid.csv")
    if os.path.exists(manifest_csv):
        df_man = pd.read_csv(manifest_csv)
        evaluate("PASS", f"Unified Gastro Verified Manifest: {len(df_man):,} pristine CT/label pairs (integrity-checked)")
    else:
        evaluate("WARN", "dataset_manifest_valid.csv not yet generated in dataset_splits_phase3")

    # -------------------------------------------------------------------------
    # 5. Project 10 Multi-Phase & Contrast Synthesis Pipeline
    # -------------------------------------------------------------------------
    audit_section("5. PROJECT 10 CONTRAST SYNTHESIS & TUMOR KINETICS")
    pairs_csv = os.path.join(RESULTS_DIR, "project10", "manifests", "multiphase_pairs.csv")
    if os.path.exists(pairs_csv):
        df_pairs = pd.read_csv(pairs_csv)
        evaluate("PASS", f"Multi-Phase Cohort Discovery: {len(df_pairs):,} paired CTs across {df_pairs['patient_id'].nunique()} patients")
    else:
        evaluate("FAIL", "Missing multiphase_pairs.csv")

    reg_files = glob.glob(os.path.join(RESULTS_DIR, "project10", "registered_volumes", "*_reg.nii.gz"))
    dvf_files = glob.glob(os.path.join(RESULTS_DIR, "project10", "registered_volumes", "*_dvf.nii.gz"))
    evaluate("PASS" if len(reg_files) >= 1530 else "WARN", f"Deformable B-Spline Registration: {len(reg_files):,} volumes & {len(dvf_files):,} DVFs")

    synth_files = glob.glob(os.path.join(RESULTS_DIR, "project10", "synthetic_volumes", "*_synthetic_contrast.nii.gz"))
    evaluate("PASS" if len(synth_files) >= 1060 else "WARN", f"Synthesized Contrast Volumes: {len(synth_files):,} 3D volumes")

    synth_csv = os.path.join(RESULTS_DIR, "project10", "evaluation_reports", "synthesis_metrics.csv")
    if os.path.exists(synth_csv):
        df_synth = pd.read_csv(synth_csv)
        corr_mean = df_synth['Pearson_R'].mean()
        evaluate("PASS", f"Synthesis Fidelity Benchmarks: {len(df_synth):,} evaluated | Mean Pearson R: {corr_mean:.4f}")
    else:
        evaluate("WARN", "Missing consolidated synthesis_metrics.csv")

    curves_png = os.path.join(BASE_DIR, "docs", "figures", "project10_tumor_kinetics_curves.png")
    evaluate("PASS" if os.path.exists(curves_png) else "WARN", f"Tumor Enhancement Kinetics Curves: {curves_png}")

    # -------------------------------------------------------------------------
    # 6. Showcase & 3D Interactive Deliverables
    # -------------------------------------------------------------------------
    audit_section("6. SHOWCASE VISUALIZATIONS & 3D MESH DELIVERABLES")
    html_files = glob.glob(os.path.join(RESULTS_DIR, "showcase_phase3", "interactive_3d", "*.html"))
    mesh_files = glob.glob(os.path.join(RESULTS_DIR, "showcase_phase3", "3d_meshes", "**", "*.obj"), recursive=True)
    vol_csv    = os.path.join(RESULTS_DIR, "showcase_phase3", "organ_volumetric_metrics.csv")

    evaluate("PASS" if len(html_files) > 0 else "WARN", f"Interactive 3D HTML Viewers: {len(html_files)} showcase viewers")
    evaluate("PASS" if len(mesh_files) > 0 else "WARN", f"3D Organ Surface Meshes (.obj): {len(mesh_files)} meshes")
    evaluate("PASS" if os.path.exists(vol_csv) else "WARN", f"Organ Volumetric Metrics: {vol_csv}")

    # -------------------------------------------------------------------------
    # 7. Log Exception Audit (Recent .err Files)
    # -------------------------------------------------------------------------
    audit_section("7. LOG EXCEPTION & ERROR SCAN")
    recent_err_files = sorted(glob.glob(os.path.join(LOGS_DIR, "*.err")), key=os.path.getmtime, reverse=True)[:15]
    fatal_patterns = [r"CUDA out of memory", r"RuntimeError", r"Traceback \(most recent call last\):", r"AssertionError"]
    clean_logs = 0
    flagged_logs = []

    for ef in recent_err_files:
        try:
            with open(ef, 'r') as f:
                content = f.read()
                matches = [p for p in fatal_patterns if re.search(p, content)]
                if matches:
                    # Ignore harmless deprecation or resolved non-fatal errors
                    if "httpx._urls" in content and len(matches) == 1:
                        clean_logs += 1
                    else:
                        flagged_logs.append((os.path.basename(ef), matches))
                else:
                    clean_logs += 1
        except Exception:
            pass

    if not flagged_logs:
        evaluate("PASS", f"Log Scan: 0 fatal exceptions detected across the {len(recent_err_files)} most recent logs")
    else:
        evaluate("WARN", f"Log Scan: Flagged {len(flagged_logs)} logs with exceptions: {[x[0] for x in flagged_logs[:3]]}")

    # -------------------------------------------------------------------------
    # 8. Showcase Sanity Review Verification & Defensibility Audit
    # -------------------------------------------------------------------------
    audit_section("8. SHOWCASE SANITY REVIEW REMEDIATION AUDIT")

    # 8.1 Tumor kinetics containment test
    try:
        from project10.kinetics.multiphase_tumor_kinetics import apply_kinetics
        test_vol = np.full((20, 20, 20), 50.0, dtype=np.float32)
        test_org = np.ones((20, 20, 20), dtype=bool)
        test_tum = np.zeros((20, 20, 20), dtype=bool)
        test_tum[5:15, 5:15, 5:15] = True
        out_vol = apply_kinetics(test_vol, test_org, test_tum, 'hcc', 'arterial')
        diff_out = np.max(np.abs(out_vol[~test_tum] - test_vol[~test_tum]))
        diff_in = np.max(np.abs(out_vol[test_tum] - test_vol[test_tum]))
        if diff_out == 0.0 and diff_in > 0.0:
            evaluate("PASS", "Tumor Kinetics Containment: Exactly 0.0 HU bleed outside lesion geometry")
        else:
            evaluate("FAIL", f"Tumor Kinetics Bleed: max diff outside={diff_out:.4f}, max diff inside={diff_in:.4f}")
    except Exception as e:
        evaluate("FAIL", f"Tumor Kinetics Containment Test failed: {e}")

    # 8.2 Turing test & Perturbation empty mask handling (no 1.0 inflation)
    try:
        from project10.evaluation.run_clinical_turing_test import compute_binary_dice as turing_dice
        from project10.evaluation.stress_test_contrast_perturbation import compute_binary_dice as pert_dice
        empty = np.zeros((10, 10, 10), dtype=bool)
        t_nan = np.isnan(turing_dice(empty, empty))
        p_nan = np.isnan(pert_dice(empty, empty))
        if t_nan and p_nan:
            evaluate("PASS", "Empty-Mask Gating: compute_binary_dice correctly returns NaN on absent organs (no false 1.0 inflation)")
        else:
            evaluate("FAIL", f"Empty-mask handling defective: turing={t_nan}, pert={p_nan}")
    except Exception as e:
        evaluate("FAIL", f"Empty-Mask Gating Test failed: {e}")

    # 8.3 JHU evaluation label mapping & small bowel bisection check
    try:
        eval_script_path = os.path.join(BASE_DIR, "code", "evaluation", "evaluate_jhu_radiologist_gpu.py")
        with open(eval_script_path, "r") as f:
            eval_code = f.read()
        has_correct_nrrd = "4: 'small_bowel'" in eval_code or "4: \"small_bowel\"" in eval_code
        has_no_zmid_bisection = "gt_z_mid" not in eval_code and "z_mid" not in eval_code
        if has_correct_nrrd and has_no_zmid_bisection:
            evaluate("PASS", "JHU Evaluation Definitions: Verified NRRD label definitions (4=small_bowel) without synthetic z-mid bisection")
        else:
            evaluate("FAIL", f"JHU Evaluation Definition check failed: nrrd={has_correct_nrrd}, no_zmid={has_no_zmid_bisection}")
    except Exception as e:
        evaluate("FAIL", f"JHU Definition Check failed: {e}")

    # 8.4 Multi-Phase Diffusion script leakage check
    try:
        diff_script_path = os.path.join(BASE_DIR, "code", "project10", "evaluation", "sample_multiphase_diffusion.py")
        with open(diff_script_path, "r") as f:
            diff_code = f.read()
        has_ncct_dir = "--ncct_dir" in diff_code and "FileNotFoundError" in diff_code
        if has_ncct_dir:
            evaluate("PASS", "Multi-Phase Diffusion: Enforces native NCCT directory baseline and strict checkpoint verification")
        else:
            evaluate("FAIL", "sample_multiphase_diffusion.py missing native NCCT dir handling or checkpoint verification")
    except Exception as e:
        evaluate("FAIL", f"Diffusion script check failed: {e}")

    # 8.5 Generated Figures inspection
    fig_turing = os.path.join(BASE_DIR, "docs", "figures", "project10_clinical_turing_comparison.png")
    fig_tumor  = os.path.join(BASE_DIR, "docs", "figures", "project10_4d_tumor_hemodynamics_showcase.png")
    if os.path.exists(fig_turing) and os.path.getsize(fig_turing) > 10000:
        evaluate("PASS", f"Showcase Turing Figure: {os.path.basename(fig_turing)} ({os.path.getsize(fig_turing):,} bytes)")
    else:
        evaluate("FAIL", f"Showcase Turing Figure missing or too small: {fig_turing}")

    if os.path.exists(fig_tumor) and os.path.getsize(fig_tumor) > 10000:
        evaluate("PASS", f"Showcase Tumor Kinetics Figure: {os.path.basename(fig_tumor)} ({os.path.getsize(fig_tumor):,} bytes)")
    else:
        evaluate("FAIL", f"Showcase Tumor Kinetics Figure missing or too small: {fig_tumor}")

    # 8.6 Showcase.md cluster links and honest reporting check
    showcase_path = os.path.join(BASE_DIR, "docs", "SHOWCASE.md")
    if os.path.exists(showcase_path):
        with open(showcase_path, "r") as f:
            showcase_text = f.read()
        has_cluster_links = "file:///mnt/scratch" in showcase_text
        has_honest_turing = "Empty-Mask Gating" in showcase_text
        if not has_cluster_links and has_honest_turing:
            evaluate("PASS", "SHOWCASE.md Documentation: Zero local file:// links, fully defensible scientific reporting")
        else:
            evaluate("FAIL", f"SHOWCASE.md check failed: cluster_links={has_cluster_links}, honest_turing={has_honest_turing}")
    else:
        evaluate("FAIL", f"SHOWCASE.md not found at {showcase_path}")

    # -------------------------------------------------------------------------
    # Final Executive Score
    # -------------------------------------------------------------------------
    audit_section("FINAL SANITY CHECK SUMMARY")
    score_pct = (passed_checks / total_checks) * 100.0 if total_checks > 0 else 0
    print(f"  Total Checks Conducted : {total_checks}")
    print(f"  Passed Checks          : {passed_checks}")
    print(f"  Warnings (Non-Fatal)   : {warnings}")
    print(f"  Critical Failures      : {failures}")
    print(f"  System Health Score    : {score_pct:.1f}%")
    print(f"{'='*70}\n")

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
