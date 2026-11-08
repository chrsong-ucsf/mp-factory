"""
code/evaluation/compute_scaling_law.py
======================================
Rigorous Statistical Scaling Law Fitting & Multi-Model Comparison.
Addresses Audit Finding 6:
- Compares Power Law, Logarithmic, and Saturating Exponential models.
- Computes parameter standard errors from covariance matrix pcov.
- Calculates R^2, RMSE, AIC, and BIC for model selection.
- Generates bootstrap 95% confidence intervals over data resamples.
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


def fit_scaling_models(csv_path: str, out_json: str = None) -> dict:
    if not os.path.exists(csv_path):
        print(f"[Error] CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    if "n_train_scans" in df.columns:
        df["train_size"] = pd.to_numeric(df["n_train_scans"], errors="coerce")
    elif "train_size" not in df.columns:
        df["train_size"] = df["cohort"].str.extract(r"N0*(\d+)").astype(float)

    clean_df = df.dropna(subset=["train_size", "mean_dice"]).copy()
    if clean_df.empty:
        print("[Error] No valid train_size and mean_dice entries found.")
        return {}

    agg_df = clean_df.groupby("train_size")["mean_dice"].agg(["mean", "std", "count"]).reset_index()
    agg_df["std"] = agg_df["std"].fillna(0.0)

    N_vals = agg_df["train_size"].values.astype(float)
    y_vals = agg_df["mean"].values.astype(float)

    results = {
        "data_points": [
            {"N": int(n), "mean_dice": round(float(y), 4), "std": round(float(s), 4), "n_obs": int(c)}
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

    # 1,000 bootstrap resamples on raw data (if multiple observations per N)
    if len(clean_df) > len(N_vals):
        boot_preds = []
        n_boot = 1000
        n_test_eval = np.linspace(min(N_vals), max(N_vals) * 1.5, 50)
        rng = np.random.RandomState(42)

        for _ in range(n_boot):
            sample = clean_df.sample(frac=1.0, replace=True, random_state=rng)
            s_agg = sample.groupby("train_size")["mean_dice"].mean().reset_index()
            try:
                popt_b, _ = curve_fit(
                    power_law, s_agg["train_size"].values, s_agg["mean_dice"].values,
                    p0=[0.85, 0.4, 0.3], bounds=([0.1, 0.0, 0.01], [1.0, 5.0, 3.0])
                )
                boot_preds.append(power_law(n_test_eval, *popt_b))
            except Exception:
                continue

        if boot_preds:
            boot_arr = np.array(boot_preds)
            results["bootstrap_ci"] = {
                "eval_N": [int(n) for n in n_test_eval],
                "pred_mean": [round(float(v), 4) for v in np.mean(boot_arr, axis=0)],
                "ci_lower_2.5": [round(float(v), 4) for v in np.percentile(boot_arr, 2.5, axis=0)],
                "ci_upper_97.5": [round(float(v), 4) for v in np.percentile(boot_arr, 97.5, axis=0)],
            }

    print(json.dumps(results, indent=2))
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved scaling law results -> {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Scaling Law Analysis")
    parser.add_argument("--csv", default="results/scaling_sweep/scaling_sweep_eval_20cases.csv", help="Input evaluation CSV")
    parser.add_argument("--out_json", default="results/scaling_law_statistical_summary.json", help="Output summary JSON")
    args = parser.parse_args()

    fit_scaling_models(args.csv, args.out_json)
