#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate saved confidence-decoder parameters (temperature + τ_c) on the validation set.

This script:
- Loads params JSON (from learn_confidence_decoder.py)
- Reorders τ_c to match the validation split's class-id mapping by name (robust to ordering)
- Applies temperature to logits, then runs *only* the confidence filtering (CF)
- Optionally applies median filter + min-seg merge on top of CF
- Reports Accuracy, F1 (macro), Edit score, and PR-AUC

By default, evaluates only epoch 100 (to mirror how params were learned).
Use --epoch all to evaluate all available validation epochs.
"""
import os, sys, json, glob, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import medfilt
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ------------------- sic-evaluation fallbacks -------------------
def get_labels_start_end_time(frame_wise_labels, bg_class=[""]):
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0]); starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i]); starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def levenstein(p, y, norm=False):
    m_row, n_col = len(p), len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1): D[i, 0] = i
    for i in range(n_col+1): D[0, i] = i
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1, j-1] + 1)
    return (1 - D[-1, -1]/max(m_row, n_col)) if norm else D[-1, -1]

def edit_score(recognized, ground_truth, norm=True, bg_class=[""]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

# ------------------------ helpers ------------------------
def softmax(x, axis=0):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

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

def reorder_tau_to_meta(action_mapping, params):
    """
    Ensure τ vector matches the current split's mapping.
    action_mapping: name->id (from logs metadata)
    params: JSON dict with 'class_names' as id->name and 'confidence_thresholds'
    """
    tau_vec = np.asarray(params["confidence_thresholds"], dtype=float)
    id2name_params = {int(k): v for k, v in params["class_names"].items()}
    # map by name to current ids
    tau_reordered = np.zeros(len(action_mapping), dtype=float)
    for name, cid in action_mapping.items():
        # find this name's id in params' space
        pid = None
        for k, v in id2name_params.items():
            if v == name:
                pid = k; break
        if pid is None or pid >= len(tau_vec):
            tau_reordered[cid] = 0.5
        else:
            tau_reordered[cid] = tau_vec[pid]
    return tau_reordered

def apply_confidence_filtering(logits_Tscaled, path, tau_vec):
    """
    logits_Tscaled: (C,T) already divided by temperature
    path: np.int array (T,)
    tau_vec: (C,)
    """
    probs = softmax(logits_Tscaled, axis=0)
    conf = probs[path, np.arange(probs.shape[1])]
    thr = np.asarray([tau_vec[c] for c in path], dtype=float)
    low = conf < thr

    out = path.copy().astype(int)
    if not np.any(low):
        return out

    idx_all = np.arange(len(out))
    hi_idx = idx_all[~low]
    if len(hi_idx) == 0:
        return out

    for i in idx_all[low]:
        j = hi_idx[np.argmin(np.abs(hi_idx - i))]
        out[i] = out[j]
    return out

def median_min_merge(labels, median_k=0, min_seg_len=0):
    """Optional smoothing on top of CF output."""
    out = labels.copy().astype(int)
    if median_k and median_k > 1 and (median_k % 2 == 1):
        out = medfilt(out, kernel_size=median_k)
    if min_seg_len and min_seg_len > 1:
        changes = np.where(np.diff(out) != 0)[0] + 1
        bounds = np.concatenate([[0], changes, [len(out)]])
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i+1]
            if (e - s) < min_seg_len:
                left = out[s-1] if s > 0 else None
                right = out[e] if e < len(out) else None
                repl = right if left is None else (left if right is None else right)
                out[s:e] = repl
    return out.astype(int)

def comprehensive_eval(y_true, y_pred, probs_for_pr=None, method=""):
    L = min(len(y_true), len(y_pred))
    y = y_true[:L]; p = y_pred[:L]
    acc = accuracy_score(y, p)
    f1m = f1_score(y, p, average="macro", zero_division=0)
    
    ed  = edit_score(p, y)
    pr_auc = 0.0
    if probs_for_pr is not None:
        try:
            from sklearn.metrics import average_precision_score
            if probs_for_pr.shape[0] != L:
                probs = probs_for_pr[:L]
            else:
                probs = probs_for_pr
            pr_auc = average_precision_score(y, probs, average="macro")
        except Exception:
            pr_auc = 0.0
    return dict(method=method, accuracy=acc, f1_score=f1m, edit_score=ed, pr_auc=pr_auc, num_frames=L, )

def save_cf_predictions(predictions, vid, ep, dataset, split, action_mapping):
    out_path = os.path.join("results_cf", dataset, f"split_{split}", vid)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    id2name = {v: k for k, v in action_mapping.items()}
    predictions = [id2name[p] for p in predictions]
    with open(out_path, "w") as f:
        f.write("### Frame level recognition: ###\n")
        f.write(" ".join(predictions))
        
# ------------------------------ main ------------------------------
def main():
    base_dir = os.path.dirname(os.path.realpath(__file__)) 
    ap = argparse.ArgumentParser(description="Evaluate confidence filtering on validation set using saved params.")
    ap.add_argument("--dataset", default="sics155-VideoMAEV2-Large-Pipeline")
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--log_dir", default=base_dir + "/probability_logs")
    ap.add_argument("--gt_dir", default=base_dir + "/data")

    ap.add_argument("--params",  default="conf_decode_params.json", help="JSON from learn_confidence_decoder.py")
    ap.add_argument("--epoch",   default="100", help="which epoch to eval: int or 'all'")
    ap.add_argument("--median_k", type=int, default=0, help="median filter kernel (odd); 0 disables")
    ap.add_argument("--min_seg",  type=int, default=0, help="merge segments shorter than this; 0 disables")
    args = ap.parse_args()

    # 1) load params
    if not os.path.exists(args.params):
        raise SystemExit(f"Params JSON not found: {args.params}")
    with open(args.params, "r") as f:
        params = json.load(f)
    Tcal = float(params.get("temperature", 1.0))

    # 2) load logs
    log_dir = os.path.join(args.log_dir, args.dataset, f"split_{args.split}")
    val_dir = os.path.join(log_dir, "validation")
    tr_dir  = os.path.join(log_dir, "train")

    train_logs = load_probability_logs(tr_dir)
    val_logs   = load_probability_logs(val_dir)
    if not train_logs: print("❌ No training logs found!"); return
    if not val_logs:   print("❌ No validation logs found!"); return

    # meta for mapping + sample_rate
    first_ep = min(train_logs.keys())
    meta = train_logs[first_ep]["metadata"]
    action_mapping = meta["action_mapping"]   # name -> id for THIS split
    sample_rate = meta.get("sample_rate", 1)

    # τ aligned to this split's ids
    tau = reorder_tau_to_meta(action_mapping, params)

    # which epochs to eval
    if str(args.epoch).lower() == "all":
        epoch_list = sorted(val_logs.keys())
    else:
        try:
            e = int(args.epoch)
            if e not in val_logs:
                raise SystemExit(f"Requested epoch {e} not in validation logs. Available: {sorted(val_logs.keys())}")
            epoch_list = [e]
        except ValueError:
            raise SystemExit("--epoch must be int or 'all'")

    # 3) run evaluation
    rows = []
    gt_root = os.path.join(args.gt_dir, args.dataset, "groundTruth")
    for ep in epoch_list:
        for vid, vdat in tqdm(val_logs[ep]["video_probabilities"].items(), desc=f"epoch {ep}"):
            gt = load_gt(os.path.join(gt_root, f"{vid}.txt"), action_mapping, sample_rate)
            if gt is None:
                continue
            logits = np.asarray(vdat["logits"])               # (C,T) "logits"
            orig   = np.asarray(vdat["predicted_classes"])    # (T,)
            Tm = min(logits.shape[1], len(gt), len(orig))
            logits = logits[:, :Tm]; orig = orig[:Tm]; gt = gt[:Tm]

            # temperature-scaled softmax (used for PR-AUC and CF confidence)
            logits_T = logits / Tcal
            probs = softmax(logits_T, axis=0).T  # (T,C) for PR-AUC

            # original
            m0 = comprehensive_eval(gt, orig, probs_for_pr=probs, method="original")
            rows.append(dict(epoch=ep, video=vid, dataset=args.dataset, split=args.split, **m0))

            # CF only
            cf = apply_confidence_filtering(logits_T, orig, tau)
            m1 = comprehensive_eval(gt, cf, probs_for_pr=probs, method="confidence_filtering")
            save_cf_predictions(cf, vid, ep, args.dataset, args.split, action_mapping)  
            rows.append(dict(epoch=ep, video=vid, dataset=args.dataset, split=args.split, **m1))

            # optional extra smoothing on top of CF
            if args.median_k or args.min_seg:
                cf_s = median_min_merge(cf, median_k=args.median_k, min_seg_len=args.min_seg)
                m2 = comprehensive_eval(gt, cf_s, probs_for_pr=probs, method="confidence_filtering+median_min")
                rows.append(dict(epoch=ep, video=vid, dataset=args.dataset, split=args.split, **m2))

    if not rows:
        print("❌ No results produced."); return

    df = pd.DataFrame(rows)
    print("\n📊 Validation (mean±std across videos)")
    agg = df.groupby(['epoch','method']).agg({
        'accuracy':['mean','std'],
        'f1_score':['mean','std'],
        'edit_score':['mean','std'],
        'pr_auc':['mean','std'],
    }).round(4)
    print(agg)

    out_csv = f"eval_confdecoder_{args.dataset}_split{args.split}_ep{','.join(map(str,epoch_list))}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved per-video results to: {out_csv}")

if __name__ == "__main__":
    main()
