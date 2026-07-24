import argparse
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


def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    ap = argparse.ArgumentParser(
        description="Write per-frame predicted action names from a FACT checkpoint to "
                    "<out_dir>/<dataset_name>/split_<split>/<video>.")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--ckpt_path", required=True, help="Path to a Checkpoint .gz to read predictions from")
    ap.add_argument("--mapping_path", default=None,
                     help="Path to mapping.txt (defaults to <this file's dir>/<dataset_name>/mapping.txt)")
    ap.add_argument("--out_dir", default="./results", help="Root output dir")
    args = ap.parse_args()

    mapping_path = args.mapping_path or os.path.join(base_dir, args.dataset_name, "mapping.txt")
    action_mapping = load_action_dict(mapping_path)
    rev_action_mapping = {v: k for k, v in action_mapping.items()}

    output_path = os.path.join(args.out_dir, args.dataset_name, f"split_{args.split}")
    os.makedirs(output_path, exist_ok=True)

    ckpt = Checkpoint.load(args.ckpt_path)
    for vname, video in tqdm(ckpt.videos.items()):
        predicted_classes = video.pred.tolist()
        with open(os.path.join(output_path, vname), "w") as f:
            f.write(" ".join(rev_action_mapping[c] for c in predicted_classes))


if __name__ == "__main__":
    main()
