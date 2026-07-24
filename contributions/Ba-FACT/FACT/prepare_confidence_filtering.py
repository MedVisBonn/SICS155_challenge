import argparse
import json
import os
from tqdm import tqdm
from .utils.evaluate import Checkpoint


def load_action_dict(mapping_path):
    file_ptr = open(mapping_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict


def dump_probabilities(ckpt, out_path, action_mapping, epoch, sample_rate):
    prob_dict = {
        "video_probabilities": {},
        "metadata": {
            "num_epochs": epoch,
            "batch_size": 1,
            "sample_rate": sample_rate,
            "action_mapping": action_mapping,
        },
    }
    for vname, video in tqdm(ckpt.videos.items()):
        prob_dict["video_probabilities"][os.path.splitext(vname)[0]] = {
            "predicted_classes": video.pred.tolist(),
            "logits": video.logit.T.tolist(),
        }
    with open(out_path, 'w') as f:
        json.dump(prob_dict, f)


def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    ap = argparse.ArgumentParser(
        description="Dump per-frame probabilities from FACT checkpoints into the JSON format "
                    "expected by learn_confidence_decoder.py.")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--train_ckpt", required=True,
                     help="Path to a Checkpoint .gz used to source train-set probabilities")
    ap.add_argument("--val_ckpt", required=True,
                     help="Path to a Checkpoint .gz used to source validation-set probabilities")
    ap.add_argument("--mapping_path", default=None,
                     help="Path to mapping.txt (defaults to <this file's dir>/<dataset_name>/mapping.txt)")
    ap.add_argument("--epoch", type=int, default=0,
                     help="Label only (used in output filenames/metadata), does not select a checkpoint")
    ap.add_argument("--sample_rate", type=int, default=1,
                     help="Dataset subsample rate (cfg.sr) -- must match the dataset's actual config "
                          "so learn_confidence_decoder.py subsamples ground truth at the right rate")
    ap.add_argument("--out_dir", default=None,
                     help="Root output dir (defaults to <this file's dir>/prob_logs)")
    args = ap.parse_args()

    mapping_path = args.mapping_path or os.path.join(base_dir, args.dataset_name, "mapping.txt")
    action_mapping = load_action_dict(mapping_path)

    out_root = args.out_dir or os.path.join(base_dir, "prob_logs")
    prob_logs_train = os.path.join(out_root, args.dataset_name, f"split_{args.split}", "train")
    prob_logs_val = os.path.join(out_root, args.dataset_name, f"split_{args.split}", "validation")
    os.makedirs(prob_logs_train, exist_ok=True)
    os.makedirs(prob_logs_val, exist_ok=True)

    ckpt_train = Checkpoint.load(args.train_ckpt)
    out_train = os.path.join(prob_logs_train, f"epoch_{args.epoch}_train_probabilities.json")
    dump_probabilities(ckpt_train, out_train, action_mapping, args.epoch, args.sample_rate)

    ckpt_val = Checkpoint.load(args.val_ckpt)
    out_val = os.path.join(prob_logs_val, f"epoch_{args.epoch}_test_probabilities.json")
    dump_probabilities(ckpt_val, out_val, action_mapping, args.epoch, args.sample_rate)


if __name__ == "__main__":
    main()
