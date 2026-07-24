"""Aggregate best-checkpoint metrics across multi-seed runs and report 95% CIs.

Usage (from repo root):
    python3 -m FACT.aggregate_seeds --logroot log/seeds_vmae_split1
"""

import argparse
import glob
import gzip
import pickle

import numpy as np
from scipy import stats

from .utils.evaluate import Checkpoint  # noqa: F401 — needed for pickle reconstruction


def _load_ckpt(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Aggregate seed-run metrics with 95% CI')
    parser.add_argument('--logroot', required=True,
                        help='Root directory that contains all seed run subdirectories')
    parser.add_argument('--n_expected', type=int, default=7,
                        help='Expected number of seed runs (for a warning if fewer found)')
    args = parser.parse_args()

    pattern = args.logroot.rstrip('/') + '/**/saves/14000.gz' # best_ckpt.gz'
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f'No best_ckpt.gz files found under {args.logroot}')
        return

    if len(files) != args.n_expected:
        print(f'WARNING: found {len(files)} checkpoint(s), expected {args.n_expected}')

    F1_KEYS = ('F1@0.10', 'F1@0.25', 'F1@0.50')

    records = []
    col_w = 72
    print(f'\n{"Run":<5}  {"Path":<{col_w}}')
    print('-' * (col_w + 8))
    for i, fpath in enumerate(files):
        ckpt = _load_ckpt(fpath)
        if not getattr(ckpt, 'metrics', None):
            print(f'{i:<5}  {fpath:<{col_w}}  [no metrics — skipping]')
            continue
        m = dict(ckpt.metrics)
        present = [m[k] for k in F1_KEYS if k in m]
        if present:
            m['F1_avg'] = float(np.mean(present))
        records.append((fpath, m))
        row = '  '.join(f'{k}={v:.2f}' for k, v in ckpt.metrics.items())
        print(f'{i:<5}  {fpath:<{col_w}}')
        print(f'       {row}')

    n = len(records)
    if n < 2:
        print('\nNeed at least 2 completed runs to compute statistics.')
        return

    keys = list(records[0][1].keys())
    t_crit = stats.t.ppf(0.975, df=n - 1)

    print(f'\n{"=" * 68}')
    print(f'Summary across {n} seeds  (95% CI: t={t_crit:.3f}, df={n-1})')
    print(f'{"=" * 68}')
    print(f'{"Metric":<14}  {"Mean":>7}  {"Std":>7}  {"±95% CI":>9}  {"Interval":>18}')
    print(f'{"-" * 68}')
    for k in keys:
        vals = np.array([m[k] for _, m in records])
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        half = t_crit * std / np.sqrt(n)
        interval = f'[{mean - half:.2f}, {mean + half:.2f}]'
        print(f'{k:<14}  {mean:>7.2f}  {std:>7.2f}  {half:>9.2f}  {interval:>18}')


if __name__ == '__main__':
    main()
