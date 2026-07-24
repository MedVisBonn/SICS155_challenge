"""Bootstrap confidence intervals for a single checkpoint's validation predictions.

Implements Method 3 ("Bootstrapping the test set predictions") from
https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-3-bootstrapping-the-test-set-predictions:
resample the validation videos of one trained model with replacement, recompute the
segmentation metrics on each resample, and take the percentile interval of the resulting
distribution. This quantifies uncertainty coming from the finite/particular set of
validation videos for a *single* run -- it does not capture across-seed/training variance
(see aggregate_seeds.py for that).

Usage (from repo root):
    python3 -m FACT.bootstrap_ci --ckpt log/<dataset>/split<n>/<exp_name>/<runid>/best_ckpt.gz
"""

import argparse
import gzip
import pickle

import numpy as np

from .utils.evaluate import Checkpoint, edit_score, f_score  # noqa: F401 -- Checkpoint needed for pickle reconstruction
from .utils.utils import parse_label

OVERLAPS = (0.10, 0.25, 0.50)
F1_KEYS = tuple(f'F1@{ov:.2f}' for ov in OVERLAPS)


def _load_ckpt(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def _per_video_stats(ckpt):
    """Precompute per-video sufficient statistics so each bootstrap resample is a cheap reduction."""
    vnames = list(ckpt.videos.keys())
    n = len(vnames)

    total = np.zeros(n)
    correct = np.zeros(n)
    fg_total = np.zeros(n)
    fg_correct = np.zeros(n)
    frame_acc = np.zeros(n)
    edit = np.zeros(n)
    tp = np.zeros((n, len(OVERLAPS)))
    fp = np.zeros((n, len(OVERLAPS)))
    fn = np.zeros((n, len(OVERLAPS)))

    for i, vname in enumerate(vnames):
        video = ckpt.videos[vname]
        gt, pred = np.asarray(video.gt_label), np.asarray(video.pred_label)

        is_correct = (gt == pred)
        fg_loc = np.array([g not in ckpt.bg_class for g in gt])

        total[i] = len(gt)
        correct[i] = is_correct.sum()
        fg_total[i] = fg_loc.sum()
        fg_correct[i] = is_correct[fg_loc].sum()
        frame_acc[i] = is_correct.mean() * 100  # per-video accuracy_score(gt_label, pred_label), matches Checkpoint's 'AccFrame'

        gt_segs = parse_label(gt)
        pred_segs = parse_label(pred)
        edit[i] = edit_score(pred_segs, gt_segs, bg_class=ckpt.bg_class)
        for j, ov in enumerate(OVERLAPS):
            tp[i, j], fp[i, j], fn[i, j] = f_score(pred_segs, gt_segs, ov, bg_class=ckpt.bg_class)

    return vnames, dict(total=total, correct=correct, fg_total=fg_total, fg_correct=fg_correct,
                         frame_acc=frame_acc, edit=edit, tp=tp, fp=fp, fn=fn)


def bootstrap(ckpt, n_boot=10000, ci=0.95, seed=None):
    ckpt.compute_metrics()  # ensure pred_label / per-video metrics are populated and consistent
    vnames, s = _per_video_stats(ckpt)
    n = len(vnames)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))  # each row = one bootstrap resample of video indices

    total = s['total'][idx].sum(axis=1)
    correct = s['correct'][idx].sum(axis=1)
    fg_total = s['fg_total'][idx].sum(axis=1)
    fg_correct = s['fg_correct'][idx].sum(axis=1)
    frame_acc = s['frame_acc'][idx].mean(axis=1)
    edit = s['edit'][idx].mean(axis=1)

    boot = {
        'AccB': correct / total * 100,
        'Acc': fg_correct / fg_total * 100,
        'AccFrame': frame_acc,
        'Edit': edit,
    }
    for j, ov in enumerate(OVERLAPS):
        tp = s['tp'][:, j][idx].sum(axis=1)
        fp = s['fp'][:, j][idx].sum(axis=1)
        fn = s['fn'][:, j][idx].sum(axis=1)
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-5) * 100
        boot[f'F1@{ov:.2f}'] = np.nan_to_num(f1)

    boot['F1_avg'] = np.mean([boot[k] for k in F1_KEYS], axis=0)

    points = dict(ckpt.metrics)
    present = [points[k] for k in F1_KEYS if k in points]
    if present:
        points['F1_avg'] = float(np.mean(present))

    alpha = (1 - ci) / 2
    results = {}
    for k, vals in boot.items():
        lo, hi = np.percentile(vals, [100 * alpha, 100 * (1 - alpha)])
        results[k] = dict(point=points.get(k), mean=float(vals.mean()),
                           std=float(vals.std(ddof=1)), lo=float(lo), hi=float(hi))
    return results, n


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI over a single checkpoint's validation videos")
    parser.add_argument('--ckpt', required=True, help='Path to a Checkpoint .gz (e.g. best_ckpt.gz) under log/')
    parser.add_argument('--n_boot', type=int, default=1000, help='Number of bootstrap resamples')
    parser.add_argument('--ci', type=float, default=0.95, help='Confidence level (percentile interval)')
    parser.add_argument('--seed', type=int, default=None, help='RNG seed for reproducibility')
    args = parser.parse_args()

    ckpt = _load_ckpt(args.ckpt)
    results, n = bootstrap(ckpt, n_boot=args.n_boot, ci=args.ci, seed=args.seed)

    print(f'\n{ckpt}  ({n} videos, {args.n_boot} bootstrap resamples, {args.ci:.0%} CI)')
    print(f'{"Metric":<10}  {"Point":>7}  {"Mean":>7}  {"Std":>7}  {"Interval":>18}')
    print('-' * 60)
    for k, r in results.items():
        point = f'{r["point"]:.2f}' if r['point'] is not None else '  n/a'
        interval = f'[{r["lo"]:.2f}, {r["hi"]:.2f}]'
        print(f'{k:<10}  {point:>7}  {r["mean"]:>7.2f}  {r["std"]:>7.2f}  {interval:>18}')


if __name__ == '__main__':
    main()
