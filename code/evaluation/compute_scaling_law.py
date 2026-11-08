"""
code/evaluation/compute_scaling_law.py
======================================
Rigorous Statistical Scaling Law Fitting & Multi-Model Comparison.
Addresses Audit Finding 6:
- Compares Power Law, Logarithmic, and Saturating Exponential models.
- Computes parameter standard errors from covariance matrix pcov.
- Calculates R^2, RMSE, AIC, and BIC for model selection.
- Patient-cluster bootstrap 95% confidence intervals on AIC-preferred model family.
- Evaluates convergence rate and parameter distributions.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# 1. Candidate Mathematical Models
def power_law(N, a, b, alpha):
    """Saturating Power Law: y = a - b * N^(-alpha)"""
    return a - b * (np.maximum(N, 1e-5) ** -alpha)


def log_model(N, a, b):
    """Logarithmic Scaling: y = a + b * ln(N)"""
    return a + b * np.log(np.maximum(N, 1e-5))


def exp_model(N, a, b, c):
    """Saturating Exponential: y = a - b * exp(-c * N)"""
    return a - b * np.exp(-c * np.maximum(N, 0.0))


def compute_metrics(y_true, y_pred, num_params):
    """Calculate R^2, RMSE, AIC, and BIC."""
    n = len(y_true)
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals ** 2))

    # Log-likelihood under Gaussian error
    sigma2 = max(ss_res / n, 1e-10)
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    k = num_params
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik

    return {
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "aic": round(float(aic), 2),
        "bic": round(float(bic), 2),
    }


def fit_scaling_models(csv_path: str, out_json: str = None, metric: str = "mean_dice", n_boot: int = 1000) -> dict:
    if not os.path.exists(csv_path):
        print(f"[Error] CSV not found: {csv_path}")
        results = {
            "status": "DATA_UNAVAILABLE",
            "message": f"Input CSV not found: {csv_path}",
            "csv_path": csv_path,
        }
        if out_json:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
        return results

    df = pd.read_csv(csv_path)

    # Resolve training size column
    if "train_scan_count" in df.columns:
        df["train_size"] = pd.to_numeric(df["train_scan_count"], errors="coerce")
    elif "n_train_scans" in df.columns:
        df["train_size"] = pd.to_numeric(df["n_train_scans"], errors="coerce")
    elif "train_size" in df.columns:
        df["train_size"] = pd.to_numeric(df["train_size"], errors="coerce")
    elif "subset_id" in df.columns:
        df["train_size"] = df["subset_id"].str.extract(r"N0*(\d+)").astype(float)
    elif "cohort" in df.columns:
        df["train_size"] = df["cohort"].str.extract(r"N0*(\d+)").astype(float)
    else:
        df["train_size"] = np.nan

    if metric not in df.columns:
        print(f"[Error] Target metric '{metric}' not found in columns: {list(df.columns)}")
        results = {"status": "ERROR", "message": f"Metric {metric} missing", "available_columns": list(df.columns)}
        return results

    clean_df = df.dropna(subset=["train_size", metric]).copy()
    if clean_df.empty or clean_df["train_size"].nunique() < 2:
        print(f"[Error] Insufficient valid data points (< 2 distinct training sizes).")
        results = {
            "status": "INSUFFICIENT_DATA",
            "message": "Fewer than 2 distinct training sizes found in CSV.",
            "csv_path": csv_path,
        }
        if out_json:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
        return results

    # Determine patient cluster column
    patient_col = None
    for cand in ["patient_id", "subject_id", "scan_id", "case_id"]:
        if cand in clean_df.columns:
            patient_col = cand
            break

    agg_df = clean_df.groupby("train_size")[metric].agg(["mean", "std", "count"]).reset_index()
    agg_df["std"] = agg_df["std"].fillna(0.0)

    N_vals = agg_df["train_size"].values.astype(float)
    y_vals = agg_df["mean"].values.astype(float)

    results = {
        "csv_path": csv_path,
        "metric": metric,
        "patient_cluster_column": patient_col,
        "n_total_observations": int(len(clean_df)),
        "n_distinct_sizes": int(len(N_vals)),
        "data_points": [
            {"N": int(n), "mean": round(float(y), 4), "std": round(float(s), 4), "n_obs": int(c)}
            for n, y, s, c in zip(N_vals, y_vals, agg_df["std"], agg_df["count"])
        ],
        "models": {},
        "preferred_model": None,
    }

    # Model 1: Power Law
    try:
        popt_p, pcov_p = curve_fit(
            power_law, N_vals, y_vals, p0=[0.85, 0.4, 0.3], bounds=([0.1, 0.0, 0.01], [1.0, 5.0, 3.0])
        )
        perr_p = np.sqrt(np.diag(pcov_p))
        y_pred_p = power_law(N_vals, *popt_p)
        met_p = compute_metrics(y_vals, y_pred_p, num_params=3)
        results["models"]["power_law"] = {
            "parameters": {"a": round(float(popt_p[0]), 4), "b": round(float(popt_p[1]), 4), "alpha": round(float(popt_p[2]), 4)},
            "param_std_err": {"a": round(float(perr_p[0]), 4), "b": round(float(perr_p[1]), 4), "alpha": round(float(perr_p[2]), 4)},
            "metrics": met_p,
        }
    except Exception as e:
        results["models"]["power_law"] = {"status": f"Fit failed: {e}"}

    # Model 2: Logarithmic
    try:
        popt_l, pcov_l = curve_fit(log_model, N_vals, y_vals, p0=[0.3, 0.08])
        perr_l = np.sqrt(np.diag(pcov_l))
        y_pred_l = log_model(N_vals, *popt_l)
        met_l = compute_metrics(y_vals, y_pred_l, num_params=2)
        results["models"]["logarithmic"] = {
            "parameters": {"a": round(float(popt_l[0]), 4), "b": round(float(popt_l[1]), 4)},
            "param_std_err": {"a": round(float(perr_l[0]), 4), "b": round(float(perr_l[1]), 4)},
            "metrics": met_l,
        }
    except Exception as e:
        results["models"]["logarithmic"] = {"status": f"Fit failed: {e}"}

    # Model 3: Saturating Exponential
    try:
        popt_e, pcov_e = curve_fit(
            exp_model, N_vals, y_vals, p0=[0.85, 0.4, 0.005], bounds=([0.1, 0.0, 1e-5], [1.0, 5.0, 1.0])
        )
        perr_e = np.sqrt(np.diag(pcov_e))
        y_pred_e = exp_model(N_vals, *popt_e)
        met_e = compute_metrics(y_vals, y_pred_e, num_params=3)
        results["models"]["exponential"] = {
            "parameters": {"a": round(float(popt_e[0]), 4), "b": round(float(popt_e[1]), 4), "c": round(float(popt_e[2]), 6)},
            "param_std_err": {"a": round(float(perr_e[0]), 4), "b": round(float(perr_e[1]), 4), "c": round(float(perr_e[2]), 6)},
            "metrics": met_e,
        }
    except Exception as e:
        results["models"]["exponential"] = {"status": f"Fit failed: {e}"}

    # Model selection via lowest AIC
    valid_models = {k: v for k, v in results["models"].items() if "metrics" in v}
    if valid_models:
        best_name = min(valid_models, key=lambda k: valid_models[k]["metrics"]["aic"])
        results["preferred_model"] = best_name
    else:
        results["preferred_model"] = None

    # Patient-Cluster Bootstrap on AIC-preferred model family
    pref = results["preferred_model"]
    if pref and len(clean_df) > len(N_vals):
        n_test_eval = np.linspace(min(N_vals), max(N_vals) * 1.5, 50)
        rng = np.random.RandomState(42)

        # Prepare clusters
        if patient_col and clean_df[patient_col].nunique() > 1:
            patient_groups = clean_df.groupby(patient_col)
            unique_entities = np.array(list(patient_groups.groups.keys()))
            use_patient_clustering = True
        else:
            unique_entities = np.arange(len(clean_df))
            use_patient_clustering = False

        model_fns = {
            "power_law": (power_law, [0.85, 0.4, 0.3], ([0.1, 0.0, 0.01], [1.0, 5.0, 3.0]), ["a", "b", "alpha"]),
            "logarithmic": (log_model, [0.3, 0.08], ([-np.inf, -np.inf], [np.inf, np.inf]), ["a", "b"]),
            "exponential": (exp_model, [0.85, 0.4, 0.005], ([0.1, 0.0, 1e-5], [1.0, 5.0, 1.0]), ["a", "b", "c"]),
        }
        target_fn, p0, bounds, param_names = model_fns[pref]

        boot_preds = []
        boot_params = []
        n_converged = 0

        for _ in range(n_boot):
            sampled_entities = rng.choice(unique_entities, size=len(unique_entities), replace=True)
            if use_patient_clustering:
                sampled_rows = pd.concat([patient_groups.get_group(e) for e in sampled_entities], ignore_index=True)
            else:
                sampled_rows = clean_df.iloc[sampled_entities]

            s_agg = sampled_rows.groupby("train_size")[metric].mean().reset_index()
            if len(s_agg) < 2:
                continue

            try:
                popt_b, _ = curve_fit(
                    target_fn, s_agg["train_size"].values, s_agg[metric].values,
                    p0=p0, bounds=bounds, maxfev=2000
                )
                pred_eval = target_fn(n_test_eval, *popt_b)
                if np.all(np.isfinite(pred_eval)):
                    boot_preds.append(pred_eval)
                    boot_params.append(popt_b)
                    n_converged += 1
            except Exception:
                continue

        convergence_rate = round(float(n_converged / n_boot), 4)

        if boot_preds:
            boot_arr = np.array(boot_preds)
            param_arr = np.array(boot_params)

            param_ci = {}
            for i, p_name in enumerate(param_names):
                param_ci[p_name] = {
                    "mean": round(float(np.mean(param_arr[:, i])), 4),
                    "std": round(float(np.std(param_arr[:, i])), 4),
                    "ci_lower_2.5": round(float(np.percentile(param_arr[:, i], 2.5)), 4),
                    "ci_upper_97.5": round(float(np.percentile(param_arr[:, i], 97.5)), 4),
                }

            results["bootstrap_ci"] = {
                "preferred_model_fitted": pref,
                "clustering_mode": "patient_cluster" if use_patient_clustering else "observation_resample",
                "cluster_entity_count": int(len(unique_entities)),
                "n_boot_requested": n_boot,
                "n_boot_converged": n_converged,
                "convergence_rate": convergence_rate,
                "parameter_ci": param_ci,
                "eval_N": [int(n) for n in n_test_eval],
                "pred_mean": [round(float(v), 4) for v in np.mean(boot_arr, axis=0)],
                "ci_lower_2.5": [round(float(v), 4) for v in np.percentile(boot_arr, 2.5, axis=0)],
                "ci_upper_97.5": [round(float(v), 4) for v in np.percentile(boot_arr, 97.5, axis=0)],
            }
        else:
            results["bootstrap_ci"] = {
                "preferred_model_fitted": pref,
                "n_boot_requested": n_boot,
                "n_boot_converged": 0,
                "convergence_rate": 0.0,
                "status": "Bootstrap failed to converge on resampled subsets.",
            }

    print(json.dumps(results, indent=2))
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved scaling law results -> {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Scaling Law Analysis with Patient-Cluster Bootstrap")
    parser.add_argument("--csv", default="results/scaling_sweep/scaling_sweep_eval_20cases.csv", help="Input evaluation CSV")
    parser.add_argument("--out_json", default="results/scaling_law_statistical_summary.json", help="Output summary JSON")
    parser.add_argument("--metric", default="mean_dice", help="Target evaluation metric column (default: mean_dice)")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap resamples (default: 1000)")
    args = parser.parse_args()

    fit_scaling_models(args.csv, args.out_json, metric=args.metric, n_boot=args.n_boot)
