from yacs.config import CfgNode as CN

_C = CN()

# auxiliary setting
_C.aux = CN()
_C.aux.gpu = 1
_C.aux.mark = "" # for adding addtional note
_C.aux.runid = 0 # the X-th run of this configuration
_C.aux.debug = False
_C.aux.wandb_project = "FACT"
_C.aux.wandb_user = ""
_C.aux.wandb_offline = False
_C.aux.resume = "max" # "", ckpt_path, "max" (resume latest ckpt of the experiment)
_C.aux.eval_every = 1000
_C.aux.print_every = 200
_C.aux.logroot = None  # override default log root (e.g., "logs/fact_sweep/")
_C.aux.save_each_eval = True  # save network.iter-XXXX.net at each eval
_C.aux.best_key = "Edit"  # metric key for selecting best checkpoint
_C.aux.seed = 194945
_C.aux.logroot = None  # override default log root (e.g., "sweep/")
_C.aux.init_from = ""

# ----- Boundary head (auxiliary) -----
_C.Boundary = Boundary = CN()
Boundary.use = False     # off by default; turn on per-experiment or via CLI
Boundary.loss_w = 0.5    # BCE weight for boundary supervision
Boundary.tv_gamma = 1.0  # exponent in (1 - p_b)^gamma for weighted TV


# dataset
_C.dataset = "breakfast"
_C.split = "split1"
_C.sr = 1 # temporal down-sample rate
_C.eval_bg = False # if including background frame in evaluation        
_C.weights = None
_C.dataset_name = None
# create dataset config variables to allow flexible dataset loading
# if configs for <_C.dataset> cannot be found at utils/dataset.py
# the code will try to read the following variables
_C.feature_path = None
_C.groundTruth_path = None
_C.map_fname = None
_C.split_path = None
_C.feature_transpose = False # if the feature is T x D or D x T
_C.bg_class = 0 # background class
_C.average_transcript_len = 0.0 # average number of segments in

# training
_C.batch_size = 4
_C.optimizer = "SGD"
_C.epoch = 2
_C.lr = 0.1
_C.lr_decay = -1
_C.momentum = 0.009
_C.weight_decay = 0.000
_C.clip_grad_norm = 10.0

_C.aux.exp = ""

#########################
# model
_C.FACT = FACT = CN()
FACT.ntoken = 30
FACT.block = "iuUU" # i - input block; u - update block; U - update block with temporal down/up-sample
FACT.trans = False # if trans is available using training + testing
FACT.fpos = True
FACT.cmr = 0.3 # channel masking rate
FACT.mwt = 0.1 # weight for merging predictions from action/frame branch

# input block
_C.Bi = Bi = CN()
Bi.hid_dim = 512
Bi.dropout = 0.5

Bi.a = "sca" 
Bi.a_nhead = 8
Bi.a_ffdim = 2048
Bi.a_layers = 6
Bi.a_dim = 512

Bi.f = 'cnn'
Bi.f_layers = 10
Bi.f_ln = True
Bi.f_dim = 512
Bi.f_ngp = 4

Bi.f_nhead = 8
Bi.f_hta_windows = [16, 64, 256]

# update block
_C.Bu = Bu = CN()
Bu.hid_dim = None
Bu.dropout = None

Bu.a = "sa"
Bu.a_nhead = None
Bu.a_ffdim = None
Bu.a_layers = 1
Bu.a_dim = None

Bu.f = None
Bu.f_layers = 5
Bu.f_ln = None
Bu.f_dim = None
Bu.f_ngp = None


Bu.f_nhead = 8
Bu.f_hta_windows = [16, 64, 256]

# update block with temporal downsample and upsample
_C.BU = BU = CN()
BU.hid_dim = None
BU.dropout = None

BU.a = "sa"
BU.a_nhead = None
BU.a_ffdim = None
BU.a_layers = 1
BU.a_dim = None

BU.f = None
BU.f_layers = 5
BU.f_ln = None
BU.f_dim = None
BU.f_ngp = None

BU.s_layers = 1

BU.f_nhead = 8
BU.f_hta_windows = [32, 128, 512]

#########################
# Loss
_C.Loss = Loss = CN()
Loss.pc = 1.0 # match weight for prob
Loss.a2fc = 1.0 # match weight for a2f_attn overlap
Loss.match = 'o2o' # one-to-one(o2o) or one-to-many(o2m)
Loss.bgw = 1.0 # weight for background class
Loss.nullw = -1.0 # weight for null class in action token; -1 -> auto-compute from statistic
Loss.sw = 0.0 # weight for smoothing loss

#########################
# temporal masking
_C.TM = TM = CN()
TM.use = False
TM.t = 30
TM.p = 0.05
TM.m = 5
TM.inplace = True

def get_cfg_defaults():
    return _C.clone()



