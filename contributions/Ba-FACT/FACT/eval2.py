import sys
import torch
from tqdm import tqdm
from .configs.utils import get_cfg_defaults
from .utils.dataset import create_dataset, DataLoader
from .utils import utils
from .utils.evaluate import Checkpoint
from .utils.train_tools import save_results


cfg = get_cfg_defaults()
cfg.merge_from_file(f'./FACT/configs/{sys.argv[4]}.yaml')
cfg.merge_from_list(sys.argv[3:])
cfg.freeze()

dataset_name = sys.argv[4]

dataset, test_dataset = create_dataset(cfg)

if dataset_name == 'epic-kitchens':
    from .models.blocks_SepVerbNoun import FACT
    model = FACT(cfg, dataset.input_dimension)
else:
    from .models.blocks import FACT
    model = FACT(cfg, dataset.input_dimension, dataset.nclasses)

weights = torch.load(cfg.weights, map_location='cpu')
if 'frame_pe.pe' in weights:
    del weights['frame_pe.pe']
model.load_state_dict(weights, strict=False)
model.eval().cuda()

ckpts = []
ckpt = Checkpoint(-1, bg_class=([] if cfg.eval_bg else dataset.bg_class))
loader = DataLoader(test_dataset, 1, shuffle=False)

for vname, batch_seq, train_label_list, eval_label in tqdm(loader):
    seq_list = [s.cuda() for s in batch_seq]
    train_label_list = [s.cuda() for s in train_label_list]
    video_saves = model(seq_list, train_label_list)
    save_results(ckpt, vname, eval_label, video_saves)

ckpt.compute_metrics()
ckpts.append(ckpt)
print(utils.easy_reduce([c.metrics for c in ckpts]))
