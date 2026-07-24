import os
import sys
import numpy as np
import torch

# Ensure local package imports work when zipped
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(THIS_DIR))
sys.path.insert(0, os.path.join(THIS_DIR, "FACT"))

from FACT.models.blocks import FACT


def _expand_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Nearest-neighbor upsample 1D label array to target length."""
    if arr.shape[0] == target_len:
        return arr
    x = torch.from_numpy(arr.astype(np.int64)).view(1, 1, -1).float()
    y = torch.nn.functional.interpolate(x, size=target_len, mode="nearest").view(-1).long()
    return y.numpy()


class _CfgNS:
    """Lightweight config namespace to mimic FACT cfg structure during inference."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def build_cfg():
    # This mirrors the effective training cfg from the finished i3d run:
    # - FACT.block iuUU, ntoken 36, cmr 0.3, mwt 0.1, no transcript
    # - Bi: a=sca (6 layers), f=MSTCN2
    FACT_cfg = _CfgNS(block="iuUU", ntoken=36, trans=False, fpos=False, cmr=0.3, mwt=0.1)
    Bi_cfg   = _CfgNS(hid_dim=512, dropout=0.0, a="sca", a_nhead=8, a_ffdim=512, a_layers=6, a_dim=512,
                      f="m2", f_layers=10, f_ln=False, f_dim=512, f_ngp=1)
    Bu_cfg   = _CfgNS(hid_dim=None, dropout=None, a="sa", a_nhead=8, a_ffdim=None, a_layers=1, a_dim=None,
                      f=None, f_layers=10, f_ln=None, f_dim=None, f_ngp=None)
    BU_cfg   = _CfgNS(hid_dim=None, dropout=None, a="sa", a_nhead=8, a_ffdim=None, a_layers=1, a_dim=None,
                      f=None, f_layers=10, f_ln=None, f_dim=None, f_ngp=None, s_layers=1)
    TM_cfg   = _CfgNS(use=True, t=30, p=0.05, m=5, inplace=True)
    Loss_cfg = _CfgNS(pc=0.2, a2fc=1.0, match="o2o", bgw=1.0, nullw=-1.0, sw=5.0)
    return _CfgNS(FACT=FACT_cfg, Bi=Bi_cfg, Bu=Bu_cfg, BU=BU_cfg, TM=TM_cfg, Loss=Loss_cfg)


class Model:
    """
    Required interface for the grader.
    - Input: dict[str -> np.ndarray[T, 1024]]
    - Output: dict[str -> np.ndarray[T] of ints]
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = 1024        # i3d feature dimension
        self.num_classes = 19        # from mapping.txt of SICS (0..18)
        self.sr = 3                 # trained with sr=2; we downsample then upsample back

        # Build minimal cfg and model
        self.cfg = build_cfg()
        self.net = FACT(self.cfg, in_dim=self.input_dim, n_classes=self.num_classes)

        # Load checkpoint placed next to this file (user should include it in the zip)
        weights_path = os.path.join(THIS_DIR, "best.net")
        state = torch.load(weights_path, map_location="cpu")
        # Tolerate positional encodings present/absent
        state.pop("frame_pe.pe", None)
        state.pop("action_pe.pe", None)
        self.net.load_state_dict(state, strict=False)
        self.net.eval().to(self.device)

    def predict(self, features: dict, video_paths: dict = None) -> dict:
        """
        features: { 'BN_0021': np.ndarray[T, 1024], ... }
        video_paths: optional dict { 'BN_0021': '/abs/path/to/video.mp4', ... } (ignored)
        returns:  { 'BN_0021': np.ndarray[T] (int labels in [0, num_classes-1]) }
        """
        outputs = {}
        with torch.no_grad():
            for vid, arr in features.items():
                assert isinstance(arr, np.ndarray) and arr.ndim == 2, f"Input for {vid} must be (T, 1024) array"

                # Ensure orientation is T x D
                x = arr
                if x.shape[1] != self.input_dim and x.shape[0] == self.input_dim:
                    x = x.T
                if x.shape[1] != self.input_dim:
                    raise ValueError(f"{vid} feature shape {x.shape} incompatible with D={self.input_dim}")

                T = x.shape[0]
                x_ds = x[::self.sr] if self.sr > 1 else x

                seq = torch.from_numpy(x_ds.astype(np.float32)).to(self.device)  # T' x 1024
                seq = seq.unsqueeze(1)  # T' x B(=1) x 1024

                # Forward and evaluate
                self.net._forward_one_video(seq)
                pred = self.net.block_list[-1].eval()  # length T'
                pred = pred.detach().cpu().numpy().astype(np.int64)

                if self.sr > 1:
                    pred = _expand_to_length(pred, T)

                outputs[vid] = pred

        return outputs


