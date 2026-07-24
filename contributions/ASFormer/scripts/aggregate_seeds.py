#!/usr/bin/env python3
"""
Aggregate per-seed evaluation logs produced by scripts/retrain_seeds.sh and
report mean +/- 95% confidence interval across seeds, for every metric
eval3.py prints (accuracy, edit score, F1@{0.10,0.25,0.50}), plus an
f1-avg metric (mean of the three F1@ scores per seed).

Follows "Method 4: Confidence Intervals From Retraining Models With
Different Random Seeds":
https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html#method-4-confidence-intervals-from-retraining-models-with-different-random-seeds

    mean = (1/r) * sum(x_j)
    SD   = sqrt( sum((x_j - mean)^2) / (r - 1) )
    SE   = SD / sqrt(r)
    CI   = mean +/- t(df=r-1, 0.975) * SE
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

METRIC_PATTERNS = {
    "acc": r"Acc:\s*([0-9.]+)",
    "edit": r"Edit:\s*([0-9.]+)",
    "f1_0.10": r"F1@0\.10:\s*([0-9.]+)",
    "f1_0.25": r"F1@0\.25:\s*([0-9.]+)",
    "f1_0.50": r"F1@0\.50:\s*([0-9.]+)",
}
METRIC_RES = {name: re.compile(pat) for name, pat in METRIC_PATTERNS.items()}
# Order metrics are displayed/written in; f1_avg is derived, not parsed directly.
METRIC_ORDER = ["acc", "edit", "f1_0.10", "f1_0.25", "f1_0.50", "f1_avg"]

FALLBACK_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042,
}


def parse_metrics(log_path):
    """Return {metric_name: value} for a single seed's log, or None if any
    of the base metrics (everything but f1_avg) is missing."""
    with open(log_path) as f:
        text = f.read()

    values = {}
    for name, regex in METRIC_RES.items():
        matches = regex.findall(text)
        if not matches:
            return None
        values[name] = float(matches[-1])

    values["f1_avg"] = (values["f1_0.10"] + values["f1_0.25"] + values["f1_0.50"]) / 3.0
    return values


def read_status(status_path):
    if not os.path.exists(status_path):
        return None
    with open(status_path) as f:
        return f.read().strip()


def discover_seeds(logdir):
    logs = glob.glob(os.path.join(logdir, "seed*.log"))
    seeds = [os.path.basename(p)[len("seed"):-len(".log")] for p in logs]
    return sorted(seeds, key=lambda s: (len(s), s))


def t_critical(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        print(f"[warn] scipy not available, using fallback t-table for df={df}", file=sys.stderr)
        return FALLBACK_T_TABLE.get(df, 1.96)


def summarize(values):
    """values: list of floats. Returns dict with r, mean, sd, se, t_crit, ci bounds."""
    r = len(values)
    mean = sum(values) / r
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (r - 1))
    se = sd / math.sqrt(r)
    df = r - 1
    t_crit = t_critical(df)
    half_width = t_crit * se
    return {
        "r": r, "mean": mean, "sd": sd, "se": se, "df": df,
        "t_crit": t_crit, "half_width": half_width,
        "ci_lower": mean - half_width, "ci_upper": mean + half_width,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default="seed_runs",
                     help="directory with seed<N>.log / seed<N>.status files (default: seed_runs)")
    ap.add_argument("--seeds", default=None,
                     help="space-separated seeds to include (default: autodiscover from --logdir)")
    args = ap.parse_args()

    seeds = args.seeds.split() if args.seeds else discover_seeds(args.logdir)
    if not seeds:
        sys.exit(f"No seed logs found in {args.logdir}")

    rows = []
    for seed in seeds:
        log_path = os.path.join(args.logdir, f"seed{seed}.log")
        status_path = os.path.join(args.logdir, f"seed{seed}.status")

        if not os.path.exists(log_path):
            print(f"[warn] missing log for seed {seed}, skipping", file=sys.stderr)
            continue

        status = read_status(status_path)
        if status is not None and status != "0":
            print(f"[warn] seed {seed} exited with status {status}, skipping", file=sys.stderr)
            continue

        metrics = parse_metrics(log_path)
        if metrics is None:
            print(f"[warn] could not parse metrics for seed {seed}, skipping", file=sys.stderr)
            continue

        rows.append((seed, metrics))

    if len(rows) < 2:
        sys.exit(f"need at least 2 successful seeds to compute a confidence interval (got {len(rows)})")

    metrics_csv = os.path.join(args.logdir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + METRIC_ORDER)
        for seed, metrics in rows:
            w.writerow([seed] + [f"{metrics[m]:.4f}" for m in METRIC_ORDER])

    stats_by_metric = {
        m: summarize([metrics[m] for _, metrics in rows]) for m in METRIC_ORDER
    }

    print(f"Per-seed metrics ({metrics_csv}):")
    header = "  seed  " + "  ".join(f"{m:>10}" for m in METRIC_ORDER)
    print(header)
    for seed, metrics in rows:
        line = f"  {seed:<6}" + "  ".join(f"{metrics[m]:>10.4f}" for m in METRIC_ORDER)
        print(line)
    print()

    r = stats_by_metric["acc"]["r"]
    df = stats_by_metric["acc"]["df"]
    print(f"r (seeds) = {r}, df = {df}")
    print()
    print(f"{'metric':<10} {'mean':>10} {'sd':>10} {'se':>10} {'t_crit':>8}   95% CI")
    for m in METRIC_ORDER:
        s = stats_by_metric[m]
        print(f"{m:<10} {s['mean']:>10.4f} {s['sd']:>10.4f} {s['se']:>10.4f} "
              f"{s['t_crit']:>8.4f}   [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}] "
              f"(+/- {s['half_width']:.4f})")

    summary_path = os.path.join(args.logdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"r={r}\n")
        f.write(f"df={df}\n")
        for m in METRIC_ORDER:
            s = stats_by_metric[m]
            f.write(f"{m}_mean={s['mean']:.4f}\n")
            f.write(f"{m}_sd={s['sd']:.4f}\n")
            f.write(f"{m}_se={s['se']:.4f}\n")
            f.write(f"{m}_t_crit={s['t_crit']:.4f}\n")
            f.write(f"{m}_ci_95_lower={s['ci_lower']:.4f}\n")
            f.write(f"{m}_ci_95_upper={s['ci_upper']:.4f}\n")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
