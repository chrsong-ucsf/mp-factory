import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json

def power_law(N, a, b, alpha):
    return a - b * (N ** -alpha)

def fit_and_print(csv_path):
    print(f"Fitting {csv_path}...")
    df = pd.read_csv(csv_path)
    
    df['train_size'] = df['cohort'].str.extract(r'N0*(\d+)').astype(float)
        
    agg_df = df.groupby('train_size')['mean_dice'].mean().reset_index()
    agg_df = agg_df.dropna()
    
    N_vals = agg_df['train_size'].values
    dice_vals = agg_df['mean_dice'].values
    
    try:
        popt, pcov = curve_fit(power_law, N_vals, dice_vals, p0=[0.8, 0.1, 0.5], bounds=([0, 0, 0], [1.0, 10, 5]))
        a, b, alpha = popt
        print(f"Fitted parameters: a={a:.4f}, b={b:.4f}, alpha={alpha:.4f}")
        return a, b, alpha
    except Exception as e:
        print("Fit failed:", e)
        return None

if __name__ == "__main__":
    fit_and_print("results/scaling_sweep/scaling_sweep_eval_20cases.csv")
