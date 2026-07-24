#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn temperature + per-class confidence thresholds (τ_c) for confidence filtering.

Defaults:
- Uses only epoch 100 (as requested).
- Learns τ_c from *raw* logits (no temperature) with a given percentile (25% by default).
- Does NOT clip τ by default (to match the monolithic trainer's behavior).
- Still learns a temperature T (applied during evaluation/inference).

You can switch to "after temperature" τ learning and enable clipping via flags.

Output JSON (default: conf_decode_params.json) fields:
  version, num_classes, class_names (id->name), temperature, confidence_thresholds,
  meta info about how they were learned (epoch list, tau_policy, percentile, clip bounds).
"""
import os, json, glob, argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import normalize


# ----------------------------- utils -----------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def load_probability_logs(log_dir):
    logs = {}
    for fp in sorted(glob.glob(os.path.join(log_dir, "epoch_*_*_probabilities.json"))):
        try:
            epoch = int(os.path.basename(fp).split("_")[1])
            with open(fp, "r") as f:
                logs[epoch] = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read {fp}: {e}")
    return logs

def load_gt(path, mapping, sample_rate=1):
    try:
        with open(path, "r") as f:
            labels = [line.strip() for line in f.read().splitlines()]
        idxs = [mapping.get(name, 0) for name in labels[::sample_rate]]
        return np.asarray(idxs, dtype=int)
    except Exception as e:
        print(f"[warn] GT load failed {path}: {e}")
        return None

def fit_temperature(train_pairs, max_samples=2_000_000, seed=0):
    # train_pairs: list of (logits[C,T], gt[T])
    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    for logits, gt in train_pairs:
        T = min(logits.shape[1], len(gt))
        Xs.append(logits[:, :T].T)  # (T,C)
        ys.append(gt[:T])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if len(y) > max_samples:
        idx = rng.choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]

    def nll(invT):
        T = max(0.05, min(5.0, 1.0 / float(invT)))
        p = softmax(X / T, axis=1)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    res = minimize_scalar(nll, bounds=(0.2, 20.0), method="bounded")
    T = max(0.05, min(5.0, 1.0 / float(res.x)))
    return T

# ------------------------------ main ------------------------------
def main():
    base_dir = os.path.dirname(os.path.realpath(__file__)) 
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sics155-VideoMAEV2-Large-Pipeline")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--run_tag", default="", help="tag appended to split dir, e.g. seed_42 (mirrors main.py --run_tag)")
    ap.add_argument("--source", choices=["train", "validation"], default="train",
                    help="which prob_logs subdirectory to fit temperature/tau from")
    ap.add_argument("--log_dir", default=base_dir + "/probability_logs")
    ap.add_argument("--gt_dir", default=base_dir + "/data")
    ap.add_argument("--epoch",   default="100", help="which epoch to use: '100' or 'all'")
    ap.add_argument("--percentile", type=float, default=25.0, help="percentile for per-class τ (0-100)")
    ap.add_argument("--tau_policy", choices=["raw", "after_temp"], default="raw",
                    help="learn τ_c from raw logits (default) or after temperature")
    ap.add_argument("--clip_tau", action="store_true", help="enable clipping τ to [tau_min, tau_max]")
    ap.add_argument("--tau_min", type=float, default=0.30)
    ap.add_argument("--tau_max", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="conf_decode_params.json")
    args = ap.parse_args()

    np.random.seed(args.seed)

    split_dir = f"split_{args.split}" + (f"_{args.run_tag}" if args.run_tag else "")
    base = os.path.join(args.log_dir, args.dataset, split_dir, args.source)
    logs = load_probability_logs(base)
    if not logs:
        raise SystemExit(f"No {args.source} logs in {base}")

    # metadata from the smallest epoch file we have
    first_ep = min(logs.keys())
    meta = logs[first_ep]["metadata"]
    action_mapping = meta["action_mapping"]           # name -> id
    inv_names = {v: k for k, v in action_mapping.items()}  # id -> name
    sample_rate = meta.get("sample_rate", 1)
    num_classes = len(action_mapping)

    # which epochs to use
    if str(args.epoch).lower() == "all":
        epoch_list = sorted(logs.keys())
    else:
        try:
            e = int(args.epoch)
            if e not in logs:
                raise SystemExit(f"Requested epoch {e} not found. Available: {sorted(logs.keys())}")
            epoch_list = [e]
        except ValueError:
            raise SystemExit("--epoch must be an int or 'all'")

    # collect (logits,gt)
    pairs = []
    print(f"[info] collecting {args.source} logits/gt for epochs {epoch_list} ...")
    for ep in epoch_list:
        for vid, vdat in tqdm(logs[ep]["video_probabilities"].items(), desc=f"epoch {ep}"):
            gt_path = os.path.join(args.gt_dir, args.dataset, "groundTruth", f"{vid}.txt")
            gt = load_gt(gt_path, action_mapping, sample_rate)
            if gt is None: 
                continue
            logits = np.asarray(vdat["logits"])  # (C,T)
            #logits = normalize(logits, norm='max', axis=0)  
            #if logits.shape[0] != num_classes:
            #    logits = logits.T  # try (T,C)
                
            Tm = min(logits.shape[1], len(gt))
            pairs.append((logits[:, :Tm], gt[:Tm]))

    if not pairs:
        raise SystemExit("No (logits,gt) pairs collected")

    # 1) temperature
    Tcal = fit_temperature(pairs, seed=args.seed)
    print(f"[ok] fitted temperature T = {Tcal:.3f}")

    # 2) per-class τ
    confs_per_class = [[] for _ in range(num_classes)]
    use_T = (args.tau_policy == "after_temp")
    for logits, gt in pairs:
        logits_use = logits / Tcal if use_T else logits
        p = softmax(logits_use, axis=0)  # (C,T)
        t = len(gt)
        idx = np.arange(t)
        conf_gt = p[gt[:t], idx]        # confidences on true class
        for c in range(num_classes):
            confs_per_class[c].extend(conf_gt[gt[:t] == c].tolist())

    tau = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        vals = np.asarray(confs_per_class[c], dtype=float)
        if vals.size == 0:
            tau[c] = 0.5
        else:
            tau[c] = np.percentile(vals, args.percentile)
        if args.clip_tau:
            tau[c] = float(np.clip(tau[c], args.tau_min, args.tau_max))

    print("[ok] per-class τ:", np.round(tau, 3))

    # 3) write JSON
    out = {
        "version": 2,
        "num_classes": num_classes,
        "class_names": inv_names,                # id -> name
        "temperature": float(Tcal),
        "confidence_thresholds": tau.tolist(),   # τ_c
        "meta": {
            "percentile": args.percentile,
            "tau_policy": args.tau_policy,
            "clip_tau": args.clip_tau,
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "epoch_list": epoch_list,
            "seed": args.seed,
            "source": args.source,
            "run_tag": args.run_tag,
        }
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved params to {args.out}\nPack this file next to model.py for submission/inference.")

if __name__ == "__main__":
    main()
