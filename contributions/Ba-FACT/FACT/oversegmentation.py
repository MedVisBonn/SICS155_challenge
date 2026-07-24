#!/usr/bin/env python3
"""Measure over-segmentation of saved predictions against ground truth.

Over-segmentation ratio for a video is (# predicted segments / # ground-truth
segments), background excluded. A ratio > 1 means the model breaks phases up
into more segments than actually occur (flickering predictions); a ratio < 1
means it merges phases together. Also reports the mean absolute difference
between predicted and ground-truth segment counts per video, a symmetric
error measure that doesn't cancel over- and under-segmentation across videos
the way the mean ratio can.

Expects predictions to already be dumped to text files in the format written
by extract_results_from_ckpt.py, i.e. one file per video at
<results_root>/<dataset_name>/split_<split>/<vname>, containing the
space-separated predicted action names.
"""

import argparse
import os


def read_bundle(path):
    with open(path, "r") as f:
        content = f.read()
    return content.splitlines()


def read_groundtruth(path):
    return read_bundle(path)


def read_recognition(path):
    with open(path, "r") as f:
        return f.read().split()


def get_labels_start_end_time(frame_wise_labels, bg_class=("background",)):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def oversegmentation_ratio(recognized, ground_truth, bg_class=("background",)):
    p_label, _, _ = get_labels_start_end_time(recognized, bg_class)
    y_label, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    if len(y_label) == 0:
        return None
    return len(p_label) / len(y_label)


def evaluate(test_ids, ground_truth_path, recog_path, bg_class=("background",)):
    ratios = []
    abs_diffs = []
    print(f"{'video':<20}{'gt_segs':>10}{'pred_segs':>12}{'ratio':>10}{'abs_diff':>10}")
    for vid in test_ids:
        gt_content = read_groundtruth(os.path.join(ground_truth_path, vid))
        recog_content = read_recognition(os.path.join(recog_path, vid))

        p_label, _, _ = get_labels_start_end_time(recog_content, bg_class)
        y_label, _, _ = get_labels_start_end_time(gt_content, bg_class)
        ratio = oversegmentation_ratio(recog_content, gt_content, bg_class)
        abs_diff = abs(len(p_label) - len(y_label))

        print(f"{vid:<20}{len(y_label):>10}{len(p_label):>12}{ratio:>10.3f}{abs_diff:>10}")
        ratios.append(ratio)
        abs_diffs.append(abs_diff)

    mean_ratio = sum(ratios) / len(ratios)
    mean_abs_diff = sum(abs_diffs) / len(abs_diffs)
    print(f"\nMean over-segmentation ratio: {mean_ratio:.4f}")
    print("(1.0 = matches ground truth, >1 = over-segmented, <1 = under-segmented)")
    print(f"Mean absolute difference in phase count: {mean_abs_diff:.4f}")
    return mean_ratio, mean_abs_diff


def parse_args():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=base_dir,
                         help="Dir containing <dataset>/{groundTruth,splits}")
    parser.add_argument("--dataset", default="sics155_c19_vmae")
    parser.add_argument("--split", default="1")
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--bg_class", nargs="*", default=["background"])
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = os.path.join(args.data_root, args.dataset)
    ground_truth_path = os.path.join(data_root, "groundTruth")
    recog_path = os.path.join(args.results_root, args.dataset, f"split_{args.split}")
    test_ids = read_bundle(os.path.join(data_root, "splits", f"test.split{args.split}.bundle"))

    evaluate(test_ids, ground_truth_path, recog_path, bg_class=tuple(args.bg_class))


if __name__ == "__main__":
    main()
