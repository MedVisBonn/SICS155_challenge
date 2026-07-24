# model.py
# =============================================================================
# Codabench submission: MS-TCN++ inference with on-the-fly VideoMAEv2-Large features
# + Confidence Filtering decoder (temperature + per-class thresholds τ_c)
# =============================================================================
# - Computes VideoMAE features from provided video_paths (offline, from local folder)
# - Z-score + L2 normalization using training stats
# - Runs MS-TCN++ (exact architecture you trained) to output framewise labels 0..18
# - Optionally applies Confidence Filtering using parameters saved in conf_decode_params.json
#
# Bundle alongside this file (flat zip root):
#   epoch-100.model
#   stats_videomae/train_mean.npy
#   stats_videomae/train_std.npy
#   conf_decode_params.json                <-- produced by learn_confidence_decoder.py
#   videomaev2_large/  (offline snapshot of OpenGVLab/VideoMAEv2-Large)
#     - include weights + preprocessor + the repo's modeling/configuration .py files
#       so transformers can load trust_remote_code offline
#
# If the worker lacks PyAV, we fall back to OpenCV for video decoding.
# If VideoMAEv2-Large cannot be loaded offline, we fallback to HF VideoMAE v1.
#
# You can tweak CF behavior via environment variables:
#   CF_DISABLE=1           -> disable confidence filtering entirely
#   CF_MEDIAN_K=7          -> optional median filter window (odd), applied after CF
#   CF_MIN_SEG=10          -> optional minimum segment merge (frames), applied after CF
# =============================================================================

import os
import sys
import math
import json
import time
import glob
import warnings
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# User-configurable constants
# ---------------------------

# Use Codabench-provided features first (I3D [T,1024]) if present?
# Set False to force computing VideoMAE features from video_paths (recommended).
USE_PROVIDED_FEATURES_FIRST = False

# Path to your locally saved OpenGVLab/VideoMAEv2-Large (offline)
# This folder must contain:
#   - pytorch_model.bin, config.json, preprocessor_config.json
#   - modeling_videomaev2.py, configuration_videomaev2.py, __init__.py (remote-code files)
LOCAL_VIDEOMAE_DIR = os.path.join(os.path.dirname(__file__), "videomaev2_large")

# Feature extraction settings (mirror your training pipeline)
TARGET_FPS   = 6            # sampled fps before upsampling back to raw
CLIP_LEN     = 16           # model.num_frames for v2-Large
CLIP_STRIDE  = 6            # stride (in sampled frames) between clip starts
BATCH_CLIPS  = 8
POOL_MODE    = "perstep"    # "perstep" (preferred for v2), "clip" for fallback
FRAMESPACE   = "raw"        # "raw" => upsample per-sampled to per-raw-frame
ALLOW_PRESAMPLE_RAW = True  # pre-sample even for framespace=raw (then upsample)
PRECISION    = "fp16"       # "fp16" on CUDA, auto-fallback to fp32 on CPU

# Memory optimization settings
CHUNK_SIZE_FRAMES = 3000    # Process video in chunks of this many sampled frames to reduce memory usage
USE_CHUNKED_READING = True  # Set False to use original non-chunked reading (for testing/comparison)

# Normalization stats (from your training run)
NORM_MEAN_PATH = os.path.join(os.path.dirname(__file__), "stats_videomae", "train_mean.npy")
NORM_STD_PATH  = os.path.join(os.path.dirname(__file__), "stats_videomae",  "train_std.npy")
APPLY_L2_AFTER_ZSCORE = True

# MS-TCN++ topology (matches your training)
FEATURES_DIM  = 1024
NUM_CLASSES   = 19
NUM_LAYERS_PG = 13
NUM_LAYERS_R  = 13
NUM_R         = 4
NUM_F_MAPS    = 96
SAMPLE_RATE   = 1           # 1 during inference (no downsample)

# Classifier weights to load (pack the file in the zip root)
WEIGHTS_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'epoch-100-96-train_plus_val.model')

# Confidence Filtering parameters JSON (pack next to model.py)
CONF_PARAMS_FILENAME = os.path.join(os.path.dirname(__file__), 'conf_decode_params.json')

# Optional post-CF smoothing taken from env (defaults: no extra smoothing)
CF_DISABLE = os.environ.get("CF_DISABLE", "0").strip() in ("1", "true", "True", "YES", "yes")
CF_MEDIAN_K = int(os.environ.get("CF_MEDIAN_K", "0"))
CF_MIN_SEG  = int(os.environ.get("CF_MIN_SEG",  "0"))

# Misc: keep torch fast
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# =============================================================================
# Utilities
# =============================================================================

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _safe_load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr)

def _l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def _standardize_provided(arr: np.ndarray) -> np.ndarray:
    """Accept [T, D] or [D, T]; return [D, T] to match MS-TCN++ input."""
    if arr.ndim != 2:
        raise ValueError(f"Provided feature array must be 2D, got {arr.shape}")
    T, D = arr.shape
    if D == FEATURES_DIM:
        return arr.T
    if T == FEATURES_DIM:
        return arr
    return arr.T

def _softmax_numpy(x: np.ndarray, axis: int = 0) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


# =============================================================================
# MS-TCN++ (exact architecture from your training file)
# =============================================================================

class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([
            Refinement(num_layers_R, num_f_maps, num_classes, num_classes) for _ in range(num_R)
        ])

    def forward(self, x):
        out = self.PG(x)                 # [B,C,T]
        outputs = out.unsqueeze(0)       # [1,B,C,T]
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs                   # [1+num_R, B, C, T]

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))
        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))
        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1) for _ in range(num_layers)
        ))
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f),
                                               self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        out = self.conv_out(f)
        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2**i, num_f_maps, num_f_maps) for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# =============================================================================
# Video reading utilities (PyAV first, OpenCV fallback)
# =============================================================================

def _open_av(video_path: str):
    try:
        import av  # PyAV
        container = av.open(video_path)
        return container
    except Exception:
        return None

def _read_frames_yield(video_path: str, target_fps: Optional[int]) -> Tuple[List[np.ndarray], List[int], Optional[int]]:
    """
    Yields RGB frames sampled at target_fps (if provided), returns:
    - frames (list of np.ndarray HxWx3),
    - sampled_raw_idx (list of raw-frame indices corresponding to sampled frames),
    - total_raw_frames (best-effort estimate)
    
    NOTE: This function loads all frames into memory. For long videos, consider using 
    _read_frames_chunked() to reduce memory usage.
    """
    container = _open_av(video_path)
    frames = []
    sampled_raw_idx = []
    total_raw = None

    if container is not None:
        try:
            stream = container.streams.video[0]
            avg_rate = float(stream.average_rate) if stream.average_rate is not None else None

            # estimate total raw frames
            if getattr(stream, "frames", 0):
                total_raw = int(stream.frames)
            elif stream.duration is not None and stream.time_base is not None and avg_rate:
                seconds = float(stream.duration * stream.time_base)
                total_raw = int(seconds * avg_rate + 0.5)

            pre_stride = 1
            if target_fps and avg_rate:
                pre_stride = max(1, int(round(avg_rate / float(target_fps))))

            idx_raw = 0
            try:
                for frame in container.decode(stream):
                    if pre_stride == 1 or (idx_raw % pre_stride == 0):
                        img = frame.to_ndarray(format="rgb24")
                        frames.append(img)
                        sampled_raw_idx.append(idx_raw)
                    idx_raw += 1
            finally:
                pass
        finally:
            container.close()
        return frames, sampled_raw_idx, total_raw

    # Fallback: OpenCV
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    pre_stride = 1
    if target_fps and fps and fps > 1e-3:
        pre_stride = max(1, int(round(float(fps) / float(target_fps))))

    idx_raw = 0
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if pre_stride == 1 or (idx_raw % pre_stride == 0):
            rgb = bgr[:, :, ::-1]
            frames.append(rgb)
            sampled_raw_idx.append(idx_raw)
        idx_raw += 1
    cap.release()
    return frames, sampled_raw_idx, total_raw


def _read_frames_chunked(video_path: str, target_fps: Optional[int], chunk_size: int = CHUNK_SIZE_FRAMES):
    """
    Generator that yields chunks of frames to reduce memory usage.
    Each chunk contains at most chunk_size frames.
    
    Yields tuples of:
    - chunk_frames: List[np.ndarray] - frames in this chunk
    - chunk_sampled_idx: List[int] - corresponding raw frame indices  
    - chunk_start_idx: int - starting sampled frame index for this chunk
    - total_raw_frames: Optional[int] - total raw frames estimate (same for all chunks)
    """
    container = _open_av(video_path)
    if container is not None:
        try:
            stream = container.streams.video[0]
            avg_rate = float(stream.average_rate) if stream.average_rate is not None else None

            # estimate total raw frames
            total_raw = None
            if getattr(stream, "frames", 0):
                total_raw = int(stream.frames)
            elif stream.duration is not None and stream.time_base is not None and avg_rate:
                seconds = float(stream.duration * stream.time_base)
                total_raw = int(seconds * avg_rate + 0.5)

            pre_stride = 1
            if target_fps and avg_rate:
                pre_stride = max(1, int(round(avg_rate / float(target_fps))))

            idx_raw = 0
            chunk_frames = []
            chunk_sampled_idx = []
            sampled_idx = 0  # Index in the sampled frames sequence
            
            try:
                for frame in container.decode(stream):
                    if pre_stride == 1 or (idx_raw % pre_stride == 0):
                        img = frame.to_ndarray(format="rgb24")
                        chunk_frames.append(img)
                        chunk_sampled_idx.append(idx_raw)
                        
                        # Yield chunk when it reaches the desired size
                        if len(chunk_frames) >= chunk_size:
                            chunk_start = sampled_idx
                            sampled_idx += len(chunk_frames)
                            yield chunk_frames, chunk_sampled_idx, chunk_start, total_raw
                            chunk_frames = []
                            chunk_sampled_idx = []
                    idx_raw += 1
                
                # Yield remaining frames in the last chunk
                if chunk_frames:
                    chunk_start = sampled_idx
                    yield chunk_frames, chunk_sampled_idx, chunk_start, total_raw
                    
            finally:
                pass
        finally:
            container.close()
        return

    # Fallback: OpenCV chunked reading
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    pre_stride = 1
    if target_fps and fps and fps > 1e-3:
        pre_stride = max(1, int(round(float(fps) / float(target_fps))))

    idx_raw = 0
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    
    chunk_frames = []
    chunk_sampled_idx = []
    sampled_idx = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if pre_stride == 1 or (idx_raw % pre_stride == 0):
            rgb = bgr[:, :, ::-1]
            chunk_frames.append(rgb)
            chunk_sampled_idx.append(idx_raw)
            
            # Yield chunk when it reaches the desired size
            if len(chunk_frames) >= chunk_size:
                chunk_start = sampled_idx
                sampled_idx += len(chunk_frames)
                yield chunk_frames, chunk_sampled_idx, chunk_start, total_raw
                chunk_frames = []
                chunk_sampled_idx = []
        idx_raw += 1
    
    # Yield remaining frames in the last chunk
    if chunk_frames:
        chunk_start = sampled_idx
        yield chunk_frames, chunk_sampled_idx, chunk_start, total_raw
    
    cap.release()


# =============================================================================
# VideoMAE feature extraction (v2 preferred, v1 fallback)
# =============================================================================

def _videomae_load():
    """
    Try to load VideoMAEv2-Large from LOCAL_VIDEOMAE_DIR (offline, trust_remote_code=True).
    If that fails, fallback to HF VideoMAE v1 (base).
    Returns:
      dict with keys:
        model, processor, kind ("v2" or "v1"),
        num_frames, tubelet, gh, gw, hidden
    """
    from transformers import AutoConfig, AutoModel, VideoMAEModel, VideoMAEImageProcessor

    # Attempt v2 (OpenGVLab) from local folder
    try:
        cfg = AutoConfig.from_pretrained(LOCAL_VIDEOMAE_DIR, trust_remote_code=True, local_files_only=True)
        processor = VideoMAEImageProcessor.from_pretrained(LOCAL_VIDEOMAE_DIR, local_files_only=True)
        model = AutoModel.from_pretrained(LOCAL_VIDEOMAE_DIR, config=cfg, trust_remote_code=True, local_files_only=True)

        device = _device()
        model.eval().to(device)

        mconf = getattr(cfg, "model_config", {}) or {}
        num_frames = int(mconf.get("num_frames", getattr(cfg, "num_frames", CLIP_LEN)))
        tubelet = int(mconf.get("tubelet_size", mconf.get("temporal_patch_size", 2)))
        patch_size = mconf.get("patch_size", 14)
        if isinstance(patch_size, (list, tuple)):
            patch_size = int(patch_size[0])
        patch_size = int(patch_size)
        hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "embed_dim", 1024)))

        in_size = 224
        gh = gw = in_size // patch_size

        return {
            "model": model, "processor": processor, "kind": "v2",
            "num_frames": num_frames, "tubelet": tubelet, "gh": gh, "gw": gw, "hidden": hidden
        }
    except Exception:
        pass  # Silently fall back to v1

    # Fallback: v1
    try:
        mid = "MCG-NJU/videomae-base"
        processor = VideoMAEImageProcessor.from_pretrained(mid, local_files_only=False)
        model = VideoMAEModel.from_pretrained(mid, local_files_only=False)

        device = _device()
        model.eval().to(device)

        num_frames = int(getattr(model.config, "num_frames", CLIP_LEN))
        hidden = int(getattr(model.config, "hidden_size", 768))

        return {
            "model": model, "processor": processor, "kind": "v1",
            "num_frames": num_frames, "tubelet": 2, "gh": 0, "gw": 0, "hidden": hidden
        }
    except Exception as e:
        raise RuntimeError(f"Cannot load any VideoMAE model: {e}")

def _to_bcthw(processor, clips_batch: List[List[np.ndarray]]) -> torch.Tensor:
    """
    Robustly call the VideoMAE processor across HF versions.
    Return pixel_values as [B,C,T,H,W].
    """
    inputs = None
    try:
        inputs = processor(videos=clips_batch, return_tensors="pt")
    except Exception:
        try:
            inputs = processor(video=clips_batch, return_tensors="pt")
        except Exception:
            inputs = None
    if inputs is None:
        # Fallback: per-clip images=...
        batch = []
        for clip in clips_batch:
            ip = processor(images=clip, return_tensors="pt")
            pv = ip["pixel_values"]  # [T,C,H,W] or [1,T,C,H,W]
            if pv.dim() == 5 and pv.shape[0] == 1:
                pv = pv.squeeze(0)
            batch.append(pv.unsqueeze(0))
        pv = torch.cat(batch, dim=0)                 # [B, T, C, H, W]
        return pv.permute(0, 2, 1, 3, 4).contiguous()

    pv = inputs["pixel_values"]                      # [B,T,C,H,W] or [B,C,T,H,W] or [B,T,H,W,C]
    if pv.dim() != 5:
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pv.shape)}")
    if pv.shape[1] in (3, 1):                        # [B,C,T,H,W]
        return pv.contiguous()
    if pv.shape[2] in (3, 1):                        # [B,T,C,H,W]
        return pv.permute(0, 2, 1, 3, 4).contiguous()
    if pv.shape[-1] in (3, 1):                       # [B,T,H,W,C]
        return pv.permute(0, 4, 1, 2, 3).contiguous()
    raise RuntimeError(f"Unexpected pixel_values layout: {tuple(pv.shape)}")

@torch.no_grad()
def _vit_tokens_v2(model, pv_bcthw: torch.Tensor) -> torch.Tensor:
    """
    Build patch tokens BEFORE global pooling for VideoMAEv2: [B, N, C]
    Assumes model.model is a ViT backbone with patch_embed and blocks.
    """
    vit = getattr(model, "model", None)
    if vit is None:
        raise RuntimeError("Unexpected v2 model: missing `.model` (ViT).")
    x = vit.patch_embed(pv_bcthw)  # [B, N, C]
    if getattr(vit, "pos_embed", None) is not None:
        x = x + vit.pos_embed.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)
    x = vit.pos_drop(x)
    for blk in vit.blocks:
        x = blk(x)
    return x  # [B, N, C]

def _tokens_to_timestep_mean(tokens: torch.Tensor, T_steps: int, gh: int, gw: int) -> torch.Tensor:
    """[B, N, C] -> [B, T_steps, gh*gw, C] -> mean over spatial -> [B, T_steps, C]."""
    B, N, C = tokens.shape
    expected = T_steps * gh * gw
    if N != expected:
        raise RuntimeError(f"Token length mismatch: got N={N}, expected {expected} (T_steps={T_steps}, gh={gh}, gw={gw}).")
    tokens = tokens.view(B, T_steps, gh * gw, C)
    return tokens.mean(dim=2)  # [B, T_steps, C]

def _aggregate_steps_to_frames(all_steps: np.ndarray, clip_starts: List[int], T_sampled: int, tubelet: int) -> np.ndarray:
    """Triangular-weighted aggregation from per-step to per-sampled-frame."""
    if all_steps.size == 0 or T_sampled <= 0:
        return np.zeros((T_sampled, 0), dtype=np.float32)
    N_clips, T_steps, D = all_steps.shape
    out = np.zeros((T_sampled, D), dtype=np.float32)
    weights = np.zeros((T_sampled,), dtype=np.float32)
    half = max(1, tubelet // 2)
    for k in range(N_clips):
        s = clip_starts[k]
        step_feats = all_steps[k]
        for j in range(T_steps):
            c = s + j * tubelet + (tubelet - 1) / 2.0
            t0 = max(0, int(math.floor(s + j * tubelet)))
            t1 = min(T_sampled, int(math.ceil(s + (j + 1) * tubelet)))
            for t in range(t0, t1):
                w = 1.0 - (abs(t - c) / (half + 1e-6))
                if w <= 0:
                    continue
                out[t] += w * step_feats[j]
                weights[t] += w
    mask = weights > 1e-8
    out[mask] /= weights[mask][:, None]
    # nearest fill
    for t in range(1, T_sampled):
        if not mask[t] and mask[t - 1]:
            out[t] = out[t - 1]; mask[t] = True
    for t in range(T_sampled - 2, -1, -1):
        if not mask[t] and mask[t + 1]:
            out[t] = out[t + 1]; mask[t] = True
    return out.astype(np.float32, copy=False)

def _aggregate_clips_to_frames(clip_feats: np.ndarray, clip_starts: List[int], T_sampled: int, clip_len: int) -> np.ndarray:
    """Coarser: one vector per clip; triangular weighting across window."""
    if clip_feats.size == 0 or T_sampled <= 0:
        return np.zeros((T_sampled, 0), dtype=np.float32)
    N, D = clip_feats.shape
    out = np.zeros((T_sampled, D), dtype=np.float32)
    weights = np.zeros((T_sampled,), dtype=np.float32)
    half = max(1, clip_len // 2)
    for k in range(N):
        s = clip_starts[k]
        e = min(T_sampled, s + clip_len)
        c = s + (clip_len - 1) / 2.0
        f = clip_feats[k]
        for t in range(s, e):
            w = 1.0 - (abs(t - c) / (half + 1e-6))
            if w <= 0:
                continue
            out[t] += w * f
            weights[t] += w
    mask = weights > 1e-8
    out[mask] /= weights[mask][:, None]
    # edge fill
    for t in range(1, T_sampled):
        if not mask[t] and mask[t - 1]:
            out[t] = out[t - 1]; mask[t] = True
    for t in range(T_sampled - 2, -1, -1):
        if not mask[t] and mask[t + 1]:
            out[t] = out[t + 1]; mask[t] = True
    return out.astype(np.float32, copy=False)

def _upsample_to_raw(per_sampled: np.ndarray, sampled_raw_idx: List[int], total_raw_frames: Optional[int]) -> np.ndarray:
    """Linear interpolation so each raw frame has a feature. per_sampled: [T_samp, D] -> [T_raw, D]."""
    if total_raw_frames is None or total_raw_frames <= 0:
        return per_sampled
    D = per_sampled.shape[1] if per_sampled.size else 0
    out = np.zeros((total_raw_frames, D), dtype=np.float32)
    if per_sampled.size == 0 or len(sampled_raw_idx) == 0:
        return out
    if len(sampled_raw_idx) == 1:
        out[:] = per_sampled[0]
        return out
    for k in range(len(sampled_raw_idx) - 1):
        i0, i1 = sampled_raw_idx[k], sampled_raw_idx[k + 1]
        f0, f1 = per_sampled[k], per_sampled[k + 1]
        span = max(1, i1 - i0)
        for i in range(i0, i1):
            a = (i - i0) / span
            out[i] = (1.0 - a) * f0 + a * f1
    first = sampled_raw_idx[0]
    if first > 0:
        out[:first] = per_sampled[0]
    last = sampled_raw_idx[-1]
    out[last:] = per_sampled[-1]
    return out


def _extract_videomae_features_from_video(video_path: str,
                                          vm: dict,
                                          target_fps: Optional[int],
                                          clip_len: int,
                                          clip_stride: int,
                                          batch_clips: int,
                                          framespace: str,
                                          precision: str) -> np.ndarray:
    """
    Returns per-frame features as [T_raw, D] if framespace='raw', else [T_samp, D].
    Uses chunked reading to reduce memory usage for long videos.
    """
    if USE_CHUNKED_READING:
        # Use chunked reading for memory efficiency
        return _extract_videomae_features_chunked(
            video_path, vm, target_fps, clip_len, clip_stride, batch_clips, framespace, precision
        )
    else:
        # Use original method (loads entire video into memory)
        return _extract_videomae_features_original(
            video_path, vm, target_fps, clip_len, clip_stride, batch_clips, framespace, precision
        )


def _extract_videomae_features_chunked(video_path: str,
                                      vm: dict,
                                      target_fps: Optional[int],
                                      clip_len: int,
                                      clip_stride: int,
                                      batch_clips: int,
                                      framespace: str,
                                      precision: str) -> np.ndarray:
    """
    Memory-efficient chunked feature extraction that produces identical results to the original.
    Processes video in chunks to reduce memory usage while maintaining exact same output.
    """
    amp = (torch.cuda.is_available() and precision.lower() in ("fp16", "bf16"))
    amp_dtype = torch.float16 if precision.lower() == "fp16" else torch.bfloat16
    device = _device()
    
    # These will accumulate results across all chunks
    all_steps_list = []
    all_clip_list = []
    all_starts_list = []
    
    # We need to track global state across chunks for consistent clip creation
    global_buffer = deque()  # Persistent buffer that spans chunks
    global_sampled_idx = 0   # Global frame index in the sampled sequence
    total_raw_frames = None
    all_sampled_raw_idx = []  # Track all raw indices for upsampling
    
    def flush_batch(clips_batch, starts_batch):
        """Process a batch of clips and extract features"""
        nonlocal all_steps_list, all_clip_list, all_starts_list
        
        if not clips_batch:
            return
            
        pv = _to_bcthw(vm["processor"], clips_batch).to(device, non_blocking=True)
        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype if amp else None):
                if vm["kind"] == "v2" and POOL_MODE == "perstep":
                    tok = _vit_tokens_v2(vm["model"], pv)          # [B, N, C]
                    T_steps = clip_len // vm["tubelet"]
                    per_step = _tokens_to_timestep_mean(tok, T_steps=T_steps, gh=vm["gh"], gw=vm["gw"])  # [B,T_steps,C]
                    all_steps_list.append(per_step.detach().cpu().numpy())
                    del tok, per_step
                else:
                    outs = vm["model"](pixel_values=pv)
                    if hasattr(outs, "last_hidden_state"):
                        x = outs.last_hidden_state.mean(dim=1)  # [B, C]
                    elif isinstance(outs, dict) and "last_hidden_state" in outs:
                        x = outs["last_hidden_state"].mean(dim=1)
                    else:
                        x = outs if torch.is_tensor(outs) else pv.mean(dim=(1, 2, 3, 4))
                    all_clip_list.append(x.detach().cpu().numpy())
                    del outs, x
        finally:
            del pv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_starts_list.extend(starts_batch)

    # Process video in chunks
    chunk_count = 0
    for chunk_frames, chunk_raw_idx, chunk_start, total_raw in _read_frames_chunked(
        video_path, target_fps if ALLOW_PRESAMPLE_RAW else None, CHUNK_SIZE_FRAMES
    ):
        if total_raw_frames is None:
            total_raw_frames = total_raw
        
        all_sampled_raw_idx.extend(chunk_raw_idx)
        
        clips_batch = []
        starts_batch = []
        
        # Add chunk frames to global buffer
        for chunk_idx, img in enumerate(chunk_frames):
            global_buffer.append(img)
            
            # Process clips from the buffer
            while len(global_buffer) >= clip_len:
                # Calculate the global start index of this clip
                s = global_sampled_idx - len(global_buffer) + 1
                clip = [global_buffer[i] for i in range(clip_len)]
                clips_batch.append(clip)
                starts_batch.append(s)
                
                # Advance the buffer by clip_stride
                for _ in range(min(clip_stride, len(global_buffer))):
                    global_buffer.popleft()
                
                # Process batch when it reaches the desired size
                if len(clips_batch) >= batch_clips:
                    flush_batch(clips_batch, starts_batch)
                    clips_batch = []
                    starts_batch = []
            
            global_sampled_idx += 1
        
        # Process any remaining clips in the batch
        if clips_batch:
            flush_batch(clips_batch, starts_batch)
        
        chunk_count += 1
        _log(f"[chunked] Processed chunk {chunk_count}, frames: {len(chunk_frames)}, buffer size: {len(global_buffer)}")

    # Handle final buffer (padding if necessary)
    if 0 < len(global_buffer) < clip_len:
        s = global_sampled_idx - len(global_buffer)
        pad = [global_buffer[-1]] * (clip_len - len(global_buffer))
        clip = list(global_buffer) + pad
        clips_batch = [clip]
        starts_batch = [s]
        flush_batch(clips_batch, starts_batch)

    global_buffer.clear()
    
    T_sampled = global_sampled_idx
    _log(f"[chunked] Total sampled frames: {T_sampled}, total clips processed: {len(all_starts_list)}")

    if T_sampled == 0:
        return np.zeros((0, vm["hidden"]), dtype=np.float32)

    # Aggregate results using the same logic as the original function
    if vm["kind"] == "v2" and POOL_MODE == "perstep" and all_steps_list:
        steps = np.concatenate(all_steps_list, axis=0)  # [N_clips, T_steps, C]
        per_sampled = _aggregate_steps_to_frames(steps, all_starts_list, T_sampled=T_sampled, tubelet=vm["tubelet"])
        del steps
        all_steps_list.clear()
    else:
        if all_clip_list:
            clips = np.concatenate(all_clip_list, axis=0)  # [N_clips, C]
            per_sampled = _aggregate_clips_to_frames(clips, all_starts_list, T_sampled=T_sampled, clip_len=clip_len)
            del clips
            all_clip_list.clear()
        else:
            per_sampled = np.zeros((T_sampled, vm["hidden"]), dtype=np.float32)

    all_starts_list.clear()

    if framespace == "raw":
        trf = total_raw_frames if total_raw_frames is not None else (all_sampled_raw_idx[-1] + 1 if all_sampled_raw_idx else per_sampled.shape[0])
        per_frame = _upsample_to_raw(per_sampled.astype(np.float32, copy=False), all_sampled_raw_idx, trf)
    else:
        per_frame = per_sampled

    return per_frame.astype(np.float32, copy=False) # [T, D]


def _extract_videomae_features_original(video_path: str,
                                       vm: dict,
                                       target_fps: Optional[int],
                                       clip_len: int,
                                       clip_stride: int,
                                       batch_clips: int,
                                       framespace: str,
                                       precision: str) -> np.ndarray:
    """
    Original implementation that loads entire video into memory.
    Kept for comparison and validation purposes.
    """
    frames, sampled_raw_idx, total_raw = _read_frames_yield(video_path, target_fps if ALLOW_PRESAMPLE_RAW else None)
    T_sampled = len(frames)

    if T_sampled == 0:
        return np.zeros((0, vm["hidden"]), dtype=np.float32)

    clips_batch: List[List[np.ndarray]] = []
    starts_batch: List[int] = []
    buffer = deque()
    steps_list = []
    clip_list = []
    starts_list = []

    amp = (torch.cuda.is_available() and precision.lower() in ("fp16", "bf16"))
    amp_dtype = torch.float16 if precision.lower() == "fp16" else torch.bfloat16
    device = _device()

    def flush_batch():
        nonlocal clips_batch, starts_batch
        if not clips_batch:
            return
        pv = _to_bcthw(vm["processor"], clips_batch).to(device, non_blocking=True)
        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp, dtype=amp_dtype if amp else None):
                if vm["kind"] == "v2" and POOL_MODE == "perstep":
                    tok = _vit_tokens_v2(vm["model"], pv)          # [B, N, C]
                    T_steps = clip_len // vm["tubelet"]
                    per_step = _tokens_to_timestep_mean(tok, T_steps=T_steps, gh=vm["gh"], gw=vm["gw"])  # [B,T_steps,C]
                    steps_list.append(per_step.detach().cpu().numpy())
                    del tok, per_step
                else:
                    outs = vm["model"](pixel_values=pv)
                    if hasattr(outs, "last_hidden_state"):
                        x = outs.last_hidden_state.mean(dim=1)  # [B, C]
                    elif isinstance(outs, dict) and "last_hidden_state" in outs:
                        x = outs["last_hidden_state"].mean(dim=1)
                    else:
                        x = outs if torch.is_tensor(outs) else pv.mean(dim=(1, 2, 3, 4))
                    clip_list.append(x.detach().cpu().numpy())
                    del outs, x
        finally:
            del pv
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        starts_list.extend(starts_batch)
        clips_batch = []
        starts_batch = []

    total_clips = 0
    for idx, img in enumerate(frames):
        buffer.append(img)
        while len(buffer) >= clip_len:
            s = (idx + 1) - len(buffer)  # sampled start index of buffer[0]
            clip = [buffer[i] for i in range(clip_len)]
            clips_batch.append(clip)
            starts_batch.append(s)
            total_clips += 1
            for _ in range(min(clip_stride, len(buffer))):
                buffer.popleft()
            if len(clips_batch) >= batch_clips:
                flush_batch()

    if 0 < len(buffer) < clip_len:
        s = T_sampled - len(buffer)
        pad = [buffer[-1]] * (clip_len - len(buffer))
        clips_batch.append(list(buffer) + pad)
        starts_batch.append(s)
        total_clips += 1
        buffer.clear()

    if clips_batch:
        flush_batch()

    buffer.clear()

    if vm["kind"] == "v2" and POOL_MODE == "perstep" and steps_list:
        steps = np.concatenate(steps_list, axis=0)  # [N_clips, T_steps, C]
        per_sampled = _aggregate_steps_to_frames(steps, starts_list, T_sampled=T_sampled, tubelet=vm["tubelet"])
        del steps
        steps_list.clear()
    else:
        if clip_list:
            clips = np.concatenate(clip_list, axis=0)  # [N_clips, C]
            per_sampled = _aggregate_clips_to_frames(clips, starts_list, T_sampled=T_sampled, clip_len=clip_len)
            del clips
            clip_list.clear()
        else:
            per_sampled = np.zeros((T_sampled, vm["hidden"]), dtype=np.float32)

    starts_list.clear()

    if framespace == "raw":
        trf = total_raw if total_raw is not None else (sampled_raw_idx[-1] + 1 if sampled_raw_idx else per_sampled.shape[0])
        per_frame = _upsample_to_raw(per_sampled.astype(np.float32, copy=False), sampled_raw_idx, trf)
    else:
        per_frame = per_sampled

    return per_frame.astype(np.float32, copy=False) # [T, D]


# =============================================================================
# Confidence Filtering (temperature + per-class thresholds)
# =============================================================================

def _load_conf_params(json_path: str, expected_num_classes: int) -> Tuple[float, np.ndarray, bool]:
    """
    Returns (temperature, tau_vec, ok_flag).
    If missing/invalid, returns (1.0, default_tau, False).
    """
    try:
        with open(json_path, "r") as f:
            params = json.load(f)
        T = float(params.get("temperature", 1.0))
        tau_list = params.get("confidence_thresholds", None)
        if tau_list is None:
            raise ValueError("confidence_thresholds missing")
        tau = np.asarray(tau_list, dtype=float)
        if tau.ndim != 1:
            raise ValueError("tau must be 1-D")
        if len(tau) != expected_num_classes:
            # If sizes differ, fallback to uniform 0.6 to be safe.
            _log(f"[warn] tau size {len(tau)} != NUM_CLASSES {expected_num_classes}; using 0.6 defaults.")
            tau = np.ones(expected_num_classes, dtype=float) * 0.6
            return T, tau, False
        return T, tau, True
    except Exception as e:
        _log(f"[info] Could not load conf params: {e}")
        return 1.0, np.ones(expected_num_classes, dtype=float) * 0.6, False

def _confidence_filtering(logits_T: np.ndarray, path: np.ndarray, tau_vec: np.ndarray) -> np.ndarray:
    """
    logits_T: (C,T) already divided by temperature
    path: (T,) integer labels from MS-TCN++
    tau_vec: (C,) per-class thresholds
    """
    probs = _softmax_numpy(logits_T, axis=0)            # (C,T)
    Tlen = probs.shape[1]
    conf = probs[path, np.arange(Tlen)]
    thr  = tau_vec[path]
    low  = conf < thr

    out = path.copy().astype(np.int64)
    if not np.any(low):
        return out

    idx_all = np.arange(Tlen, dtype=np.int64)
    hi_idx = idx_all[~low]
    if len(hi_idx) == 0:
        return out  # no confident frames at all -> leave as-is

    # nearest confident frame fill
    for i in idx_all[low]:
        j = hi_idx[np.argmin(np.abs(hi_idx - i))]
        out[i] = out[j]
    return out

def _median_min_merge(labels: np.ndarray, median_k: int = 0, min_seg_len: int = 0) -> np.ndarray:
    """Optional smoothing after CF."""
    out = labels.copy().astype(np.int64)

    # Median filter
    if median_k and median_k > 1 and (median_k % 2 == 1):
        # Simple 1D median; implement minimal dependency to keep the file self-contained
        # (No SciPy here; rolling median)
        k = median_k
        pad = k // 2
        padded = np.pad(out, (pad, pad), mode='edge')
        tmp = np.empty_like(out)
        for i in range(len(out)):
            tmp[i] = np.median(padded[i:i+k])
        out = tmp.astype(np.int64)

    # Merge too-short segments
    if min_seg_len and min_seg_len > 1:
        changes = np.where(np.diff(out) != 0)[0] + 1
        bounds = np.concatenate([[0], changes, [len(out)]])
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if (e - s) < min_seg_len:
                left  = out[s - 1] if s > 0 else None
                right = out[e] if e < len(out) else None
                # choose neighbor (prefer right for consistency)
                repl = right if left is None else (left if right is None else right)
                out[s:e] = repl
    return out


# =============================================================================
# The submission Model class
# =============================================================================

class Model:
    def __init__(self):
        """Initialize the classifier, VideoMAE extractor, and CF parameters."""
        self.name = "MS-TCN2-VideoMAEv2-Submission+CF"
        self.device = _device()

        # ---- load normalization stats (for VideoMAE features) ----
        if os.path.exists(NORM_MEAN_PATH) and os.path.exists(NORM_STD_PATH):
            self.norm_mean = _safe_load_npy(NORM_MEAN_PATH).astype(np.float32)
            self.norm_std  = _safe_load_npy(NORM_STD_PATH).astype(np.float32)
        else:
            self.norm_mean = None
            self.norm_std  = None

        # ---- build MS-TCN++ classifier & load weights ----
        self.classifier = MS_TCN2(
            num_layers_PG=NUM_LAYERS_PG,
            num_layers_R=NUM_LAYERS_R,
            num_R=NUM_R,
            num_f_maps=NUM_F_MAPS,
            dim=FEATURES_DIM,
            num_classes=NUM_CLASSES,
        ).to(self.device).eval()

        weights_path = WEIGHTS_MODEL_FILENAME
        if not os.path.exists(weights_path):
            candidates = glob.glob("**/" + os.path.basename(WEIGHTS_MODEL_FILENAME), recursive=True)
            if candidates:
                weights_path = candidates[0]
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        sd = torch.load(weights_path, map_location=self.device)
        try:
            self.classifier.load_state_dict(sd, strict=True)
        except Exception:
            self.classifier.load_state_dict(sd, strict=False)

        # ---- try to prepare VideoMAE extractor (v2 -> v1 fallback) ----
        try:
            self.vmae = _videomae_load()
            _log(f"[info] VideoMAE model loaded: {self.vmae['kind'] if self.vmae else 'None'}")
        except Exception:
            self.vmae = None

        # ---- load Confidence Filtering parameters ----
        self.T_cal, self.tau_vec, self.cf_params_ok = _load_conf_params(CONF_PARAMS_FILENAME, NUM_CLASSES)
        if CF_DISABLE:
            _log("[info] CF disabled via CF_DISABLE env.")
            self.cf_params_ok = False
        else:
            if self.cf_params_ok:
                _log(f"[info] CF ready. T={self.T_cal:.3f}, tau[0:5]={np.round(self.tau_vec[:5],3)}")
            else:
                _log("[info] CF params not available; using raw model outputs.")
        
        # Log memory optimization settings
        if USE_CHUNKED_READING:
            _log(f"[info] Memory optimization: Chunked reading enabled (chunk size: {CHUNK_SIZE_FRAMES} frames)")
        else:
            _log("[info] Memory optimization: Using original full-video loading (higher memory usage)")

    def fit(self):
        """No-op: model is already trained."""
        return

    def cleanup(self):
        """Explicitly clean up GPU memory and model resources."""
        if hasattr(self, 'classifier') and self.classifier is not None:
            self.classifier.cpu()
        if hasattr(self, 'vmae') and self.vmae is not None and 'model' in self.vmae:
            self.vmae['model'].cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _log("[cleanup] Released GPU memory and moved models to CPU.")

    # -----------------------------
    # Prediction helper functions
    # -----------------------------

    def _normalize_like_training(self, x_TxD: np.ndarray) -> np.ndarray:
        """Apply train mean/std and L2 if configured. x_TxD is [T, D]."""
        if self.norm_mean is None or self.norm_std is None:
            return x_TxD.astype(np.float32, copy=False)
        std = np.where(self.norm_std < 1e-8, 1.0, self.norm_std).astype(np.float32, copy=False)
        y = (x_TxD - self.norm_mean[None, :]) / std[None, :]
        if APPLY_L2_AFTER_ZSCORE:
            y = _l2_normalize(y.astype(np.float32, copy=False))
        return y

    def _get_features_for_video(self, key: str, provided: Dict[str, np.ndarray], vpath: Optional[str]) -> np.ndarray:
        """
        Returns features as [D, T] ready for MS-TCN++.
        If USE_PROVIDED_FEATURES_FIRST and present -> uses them (transpose if needed).
        Else -> compute VideoMAE features from vpath.
        """
        if USE_PROVIDED_FEATURES_FIRST and key in provided:
            arr = provided[key]
            x_DxT = _standardize_provided(np.asarray(arr))
            if x_DxT.shape[0] != FEATURES_DIM:
                raise RuntimeError(f"Provided features for {key} have dim={x_DxT.shape[0]} but FEATURES_DIM={FEATURES_DIM}.")
            return x_DxT.astype(np.float32, copy=False)

        if not vpath or not os.path.exists(vpath):
            raise RuntimeError(f"No video_path for {key}, and provided features are disabled/missing.")
        if self.vmae is None:
            raise RuntimeError("VideoMAE extractor is not available.")

        x_TxD = _extract_videomae_features_from_video(
            vpath, self.vmae,
            target_fps=(TARGET_FPS if (FRAMESPACE == "raw" and ALLOW_PRESAMPLE_RAW) or FRAMESPACE == "sampled" else None),
            clip_len=CLIP_LEN,
            clip_stride=CLIP_STRIDE,
            batch_clips=BATCH_CLIPS,
            framespace=FRAMESPACE,
            precision=PRECISION,
        )

        x_TxD = self._normalize_like_training(x_TxD)
        x_DxT = x_TxD.astype(np.float32, copy=False).T

        if x_DxT.shape[0] != FEATURES_DIM:
            raise RuntimeError(f"Extracted dim={x_DxT.shape[0]} but FEATURES_DIM={FEATURES_DIM}. "
                               f"(Did you train on a different backbone?)")
        return x_DxT

    # -----------------------------
    # Codabench prediction interface
    # -----------------------------

    def predict(self, features: Dict[str, np.ndarray], video_paths: Dict[str, str] = None) -> Dict[str, List[int]]:
        """
        Predict labels from features (or compute from video_paths).
        - features: dict vid -> np.ndarray [T, 1024] (Codabench I3D) OR [D, T]; we handle both.
        - video_paths: dict vid -> absolute path to .mp4; used to compute VideoMAE features.
        Return: dict vid -> list[int] of length T (framewise labels 0..18).
        """
        video_paths = video_paths or {}
        keys = sorted(set(features.keys()) | set(video_paths.keys()))
        if not keys:
            return {}

        out_preds: Dict[str, List[int]] = {}

        with torch.inference_mode():
            for i, key in enumerate(keys):
                try:
                    # 1) Get features [D, T]
                    vpath = video_paths.get(key, None)
                    x_DxT = self._get_features_for_video(key, features, vpath)

                    # 2) (optional) sample rate (kept at 1 to mirror training)
                    if SAMPLE_RATE > 1:
                        x_DxT = x_DxT[:, ::SAMPLE_RATE]

                    # 3) Run classifier
                    x = torch.from_numpy(x_DxT).unsqueeze(0).to(self.device)  # [1, D, T]
                    preds_all = self.classifier(x)                            # [1+R, 1, C, T]
                    logits = preds_all[-1]                                    # [1, C, T]
                    _, pred = torch.max(logits, dim=1)                        # [1, T]
                    labels = pred.squeeze(0).detach().cpu().numpy().astype(np.int64)  # (T,)

                    # 4) Confidence Filtering (if params loaded and not disabled)
                    if self.cf_params_ok:
                        logits_np = logits.squeeze(0).detach().cpu().numpy()        # (C,T)
                        logits_T = logits_np / float(self.T_cal)                    # apply temperature
                        cf_labels = _confidence_filtering(logits_T, labels, self.tau_vec)
                        if CF_MEDIAN_K or CF_MIN_SEG:
                            cf_labels = _median_min_merge(cf_labels, CF_MEDIAN_K, CF_MIN_SEG)
                        final_labels = cf_labels.astype(np.int64, copy=False)
                    else:
                        final_labels = labels

                    out_preds[key] = final_labels.tolist()

                    # Clean up tensors/arrays
                    del x, preds_all, logits, pred, x_DxT, labels
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    remaining = len(keys) - (i + 1)
                    _log(f"Completed {key} | {i+1}/{len(keys)} | Remaining: {remaining}")

                except Exception as e:
                    _log(f"[error] {key}: {e}")
                    out_preds[key] = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return out_preds
