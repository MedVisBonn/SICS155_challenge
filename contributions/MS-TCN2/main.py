#!/usr/bin/python2.7

import json
import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)


parser.add_argument('--num_f_maps', default='96', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)
parser.add_argument('--sample_rate', type=int, default=1)
parser.add_argument('--lambda_val', type=float, default=0.35)
parser.add_argument('--seed', type=int, default=1538574472)
parser.add_argument('--run_tag', default='', help='tag appended to output dirs, e.g. seed_42')

args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = args.sample_rate
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

split_dir = "split_" + args.split + ("_" + args.run_tag if args.run_tag else "")

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

model_dir = "./models/"+args.dataset+"/"+split_dir
results_dir = "./results/"+args.dataset+"/"+split_dir
prob_logs_dir = "./prob_logs/"+args.dataset+"/"+split_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(prob_logs_dir):
    os.makedirs(prob_logs_dir)
    os.mkdir(os.path.join(prob_logs_dir, "train"))
    os.mkdir(os.path.join(prob_logs_dir, "validation"))

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

prob_logs = None
num_classes = len(actions_dict)
trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split, args.lambda_val)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    prob_logs = trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    prob_logs = trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)

subdir = "train" if args.action == "train" else "validation"
prob_logs["metadata"]["action_mapping"] = actions_dict
with open(os.path.join(prob_logs_dir, subdir, f'epoch_{args.num_epochs}_{subdir}_probabilities.json'), 'w') as f:
        json.dump(prob_logs, f)

