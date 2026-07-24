#!/usr/bin/env python3
"""
Print mean, 95% CI, and per-run values for each metric across uncertainty runs.
Uses Method 4 from Raschka (2022): t-distribution CI from runs with different seeds.

Usage:
    python summarize_uncertainty.py <metrics.tsv>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

METRIC_LABELS = {
    'acc':    'Acc',
    'edit':   'Edit',
    'f1_10':  'F1@0.10',
    'f1_25':  'F1@0.25',
    'f1_50':  'F1@0.50',
    'f1_avg': 'F1@avg',
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    df['f1_avg'] = df[['f1_10', 'f1_25', 'f1_50']].mean(axis=1)
    return df


def compute_ci95(vals: np.ndarray) -> tuple[float, float, float, float]:
    """CI = mean ± t_{r-1, 0.975} × (sample_std / sqrt(r)), ddof=1."""
    r = len(vals)
    mean = vals.mean()
    half_width = t_dist.ppf(0.975, df=r - 1) * vals.std(ddof=1) / np.sqrt(r)
    return mean, half_width, mean - half_width, mean + half_width


def format_values(vals: np.ndarray) -> str:
    return "  ".join(f"{v:.2f}" for v in vals)


def print_table(df: pd.DataFrame, path: str) -> None:
    metric_cols = [c for c in df.columns if c != 'seed']
    print(f"\n{'Metric':<12} {'Mean':>8}  {'CI95±':>8}  {'[CI Low,  CI High]':<21}  Values")
    print("-" * 80)
    for col in metric_cols:
        vals = df[col].to_numpy()
        mean, half_width, ci_low, ci_high = compute_ci95(vals)
        label = METRIC_LABELS.get(col, col)
        print(
            f"{label:<12} {mean:>8.4f}  ±{half_width:>6.4f}"
            f"  [{ci_low:>8.4f}, {ci_high:>8.4f}]  {format_values(vals)}"
        )
    seeds = df['seed'].astype(str).tolist()
    print(f"\nRuns: {len(df)}  |  Seeds: {', '.join(seeds)}")
    print(f"Results from: {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python summarize_uncertainty.py <metrics.tsv>")
        sys.exit(1)
    path = sys.argv[1]
    df = load_data(path)
    if df.empty:
        print(f"No data found in {path}")
        sys.exit(1)
    print_table(df, path)


if __name__ == '__main__':
    main()
