# Confidence Filtering Pipeline

How to apply the confidence filter to a FACT model's predictions after a training run.

Training already saves `Video.logit` (per-frame blended softmax probability) into every
checkpoint automatically — no extra flags needed.

## 1. Train

```bash
python3 -m FACT.train --cfg configs/your_config.yaml --set aux.gpu 0 split "split1"
```

This produces `log/<dataset>/<split>/<exp_name>/<runid>/last_ckpt.gz` and `best_ckpt.gz`.

## 2. Dump per-frame probabilities to JSON

`train.py` already exercises `net(...)` at train time, so any checkpoint from step 1 has
`.logit` populated. Pick one checkpoint to represent "train" and one for "validation" (e.g. both
from the same run):

```bash
python3 -m FACT.prepare_confidence_filtering \
  --dataset_name <dataset> --split <n> \
  --train_ckpt log/<dataset>/split<n>/<exp_name>/<runid>/last_ckpt.gz \
  --val_ckpt   log/<dataset>/split<n>/<exp_name>/<runid>/best_ckpt.gz \
  --epoch <label> --sample_rate <cfg.sr>
```

Writes to `FACT/prob_logs/<dataset>/split_<n>/{train,validation}/epoch_<label>_*.json` by default.

## 3. Learn the temperature + per-class thresholds

Learned from the train-split probabilities:

```bash
python3 learn_confidence_decoder.py --dataset <dataset> --split <n> --epoch <label> --out conf_decode_params.json
```

Defaults for `--log_dir`/`--gt_dir` line up automatically with step 2's output and the dataset's
real ground-truth folder.

## 4. Apply the filter

```bash
python3 eval_confidence_decoder.py --dataset <dataset> --split <n> --params conf_decode_params.json --epoch <label>
```

This is the actual "application" step — it temperature-scales the probabilities, replaces
low-confidence frames with the nearest high-confidence prediction, writes filtered per-frame
predictions to `results_cf/<dataset>/split_<n>/<video>`, and prints/saves an accuracy/F1/edit-score/
PR-AUC comparison of `original` vs `confidence_filtering`.

Run this from the repo root — `results_cf/` and the metrics `.csv` are written relative to the
current working directory, not repo-root or script-relative.

## Background: why the logit field needs a log-transform

`Video.logit` (and the `"logits"` key in the JSON produced by step 2) is not a raw pre-softmax
logit — it's the blended **post-softmax probability** that already determines `pred`
(`pred = prob.argmax(1)`), values in `[0,1]` summing to ~1 per frame. This was a deliberate choice
(purely additive, `pred`/`Acc`/`Edit`/`F1` unaffected) over reverting to an old raw-logit-blend
implementation.

`learn_confidence_decoder.py` and `eval_confidence_decoder.py` were originally written assuming
this field holds true raw logits (confirmed by inspecting old JSON dumps with unbounded negative
values). Feeding probabilities through their softmax/temperature-scaling code directly would
silently double-apply softmax, crushing confidence values toward uniform and defeating calibration
— without ever crashing or changing `pred`, so it's easy to miss.

Both scripts now log-transform the loaded array first: `logits = np.log(logits + 1e-12)`. This
works because for `p = softmax(L)`, `log(p) = L - c` for a per-frame constant `c`, and softmax is
invariant to adding an equal constant to every class score, so `softmax(log(p)/T) == softmax(L/T)`
for any `T` — i.e. `log(p)` is a fully valid drop-in replacement for the true logits for every
downstream operation these scripts perform.
