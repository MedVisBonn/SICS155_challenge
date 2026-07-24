# FACT: Frame-Action Cross-Attention Temporal Segmentation

## Overview

FACT is a transformer-based architecture for temporal action segmentation that models the relationship between frame-level features and action-level representations using cross-attention mechanisms.

## Architecture Components

### Core Modules

The FACT model consists of three main block types arranged in a sequence specified by the `block` configuration parameter:

1. **Input Block (i)**: Processes raw features and initializes action queries
2. **Update Block (u)**: Refines features through bidirectional frame-action communication
3. **Update Block with Temporal Downsampling (U)**: Operates at segment-level for efficiency

### Block Architecture

Each block contains:
- **Frame Branch**: Temporal convolution layers (MSTCN/MSTCN2) for frame-level processing
- **Action Branch**: Self-attention or cross-attention layers for action-level reasoning
- **Cross-Attention Layers**: Bidirectional communication between frames and actions
  - `f2a_layer`: Frame-to-Action attention
  - `a2f_layer`: Action-to-Frame attention

### Boundary Head Extension

The boundary head is an optional component that predicts frame-level boundary probabilities to improve temporal smoothness and segmentation quality.

#### Implementation Details

Location: `FACT/models/blocks.py`

Each block can optionally include a boundary head that:
- Predicts binary boundary probabilities at each frame
- Uses a 1D convolutional layer: `Conv1d(H_feat, 1, kernel_size=3, padding=1)`
- Outputs sigmoid probabilities indicating likely action boundaries

**Enabling Boundary Head:**

Add to configuration:
```yaml
Boundary:
  use: True           # Enable boundary prediction
  tv_gamma: 1.0       # Gamma exponent for boundary-weighted smoothing
  loss_w: 0.5         # Weight for boundary BCE loss
```

**Code Structure:**

```python
# In each Block class (InputBlock, UpdateBlock, UpdateBlockTDU):

def _enable_boundary_head(self):
    """Set flag; boundary head is lazily created with correct dimensions."""
    self.boundary_on = True
    self.boundary_head = None

def _boundary_forward(self, prelogit_frame_feat, nclass):
    """
    Forward pass through boundary head.
    
    Args:
        prelogit_frame_feat: T×B×Hpre (frame features before logit extraction)
        nclass: Number of action classes
        
    Returns:
        bprob: T×B×1 boundary probabilities (sigmoid)
    """
    if not getattr(self, 'boundary_on', False):
        return None
    
    H_feat = prelogit_frame_feat.shape[-1] - nclass
    x = prelogit_frame_feat[:, :, :H_feat].permute(1, 2, 0)  # B×H×T
    
    # Lazy initialization
    if self.boundary_head is None or self.boundary_head.in_channels != H_feat:
        self.boundary_head = nn.Conv1d(H_feat, 1, kernel_size=3, padding=1)
        self.boundary_head.to(prelogit_frame_feat.device)
    
    b = torch.sigmoid(self.boundary_head(x))  # B×1×T
    return b.permute(2, 0, 1)  # T×B×1
```

**Loss Computation:**

Location: `FACT/models/loss.py`

Two boundary-related loss components:

1. **Boundary-Weighted Temporal Smoothing:**
```python
def smooth_loss_weighted(logit, bprob, gamma=1.0, is_logit=True):
    """
    Applies temporal smoothing with reduced weight at predicted boundaries.
    
    Weight at each frame: (1 - p_boundary)^gamma
    - High boundary probability → low smoothing weight → allows transitions
    - Low boundary probability → high smoothing weight → encourages stability
    """
```

2. **Boundary Binary Cross-Entropy:**
```python
def boundary_bce(pred_b, gt_b):
    """
    Supervises boundary predictions against ground-truth label changes.
    
    gt_b[t] = 1 if label[t] != label[t-1], else 0
    """
```

**Usage in Block Loss:**

```python
def compute_loss(self, criterion, match=None):
    frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
    atk_loss = criterion.action_token_loss(match, self.action_clogit)
    
    frame_clogit = torch.transpose(self.frame_clogit, 0, 1)
    
    if cfg.Boundary.use and (self.bprob is not None):
        # Boundary-weighted smoothing
        smooth_loss = loss.smooth_loss_weighted(
            frame_clogit, self.bprob, 
            gamma=cfg.Boundary.tv_gamma
        )
        # Boundary prediction loss
        bce = loss.boundary_bce(
            self.bprob, 
            loss.boundary_targets_from_label(criterion.class_label)
        )
        return frame_loss + atk_loss + cfg.Loss.sw * smooth_loss + cfg.Boundary.loss_w * bce
    else:
        # Standard smoothing
        smooth_loss = loss.smooth_loss(frame_clogit)
        return frame_loss + atk_loss + cfg.Loss.sw * smooth_loss
```

#### Impact on Training vs Inference

**Training:**
- Boundary head adds supervision signal for detecting action transitions
- Weighted smoothing loss preserves sharpness at true boundaries
- Improves temporal localization of action segments

**Inference:**
- Current implementation: boundary predictions are computed but not used in final prediction
- Predictions come from `Block.eval()` which combines action and frame branch logits
- Future enhancement: boundary probabilities could guide post-processing or adaptive smoothing

## Dataset Setup

See `DATASET_SETUP_GUIDE.md` for detailed instructions on preparing datasets.

### Quick Start

1. Organize your dataset:
```
dataset_name/
├── features/           # Video features (.npy files, shape: T×D)
├── groundTruth/       # Frame labels (.txt files)
├── mapping.txt        # Class ID to name mapping
└── splits/            # Train/test split files (.bundle)
```

2. Generate configuration:
```bash
python utils/gen_config.py \
    --dataset_path /path/to/dataset \
    --dataset_name dataset_name \
    --output_config configs/dataset_name.yaml \
    --base_config configs/breakfast.yaml
```

3. Verify configuration parameters:
```yaml
dataset: dataset_name
dataset_path: /path/to/dataset
feature_size: 2048  # Match your feature dimension
num_classes: 10     # Number of action classes
splits: ['1', '2', '3', '4']
```

## Training

### Basic Training

Train a single model with specified configuration:

```bash
python -m FACT.train --cfg configs/your_config.yaml
```

#### Example: Single Training Run

See `jobs/fact_train.sh` for a complete example that:
- Sets up environment and W&B configuration
- Generates dataset config from template
- Launches training in background with nohup
- Logs output to timestamped file

#### Training with Boundary Head

Enable boundary head with additional parameters:

```bash
python3 -m FACT.train \
  --cfg FACT/configs/sics3.yaml \
  --set Boundary.use true \
       Boundary.loss_w 1.0 \
       Boundary.tv_gamma 2.0 \
       Loss.sw 5.0 \
       aux.gpu 0
```

Key configuration parameters:

```yaml
# Model Architecture
FACT:
  block: "iuUU"        # Block sequence: i=input, u=update, U=update+TDU
  ntoken: 36           # Number of action query tokens
  trans: False         # Use transcript (False for no transcript)
  fpos: False          # Use frame positional encoding
  cmr: 0.3            # Channel masking rate
  mwt: 0.5            # Merge weight between action/frame branches

# Training
optimizer: Adam
lr: 0.0003             # Learning rate
batch_size: 8
epoch: 100             # Number of training epochs
sr: 3                  # Temporal sample rate (downsample factor)
lr_decay: 120          # Learning rate decay epoch

# Loss
Loss:
  match: o2m           # Matching strategy: o2o (one-to-one) or o2m (one-to-many)
  sw: 2.5             # Smoothness loss weight
  pc: 0.2             # Prediction cost weight
  a2fc: 1.0           # Action-to-frame cost weight
  bgw: 1.0            # Background class weight
  nullw: -1.0         # Null token weight

# Boundary Head (optional)
Boundary:
  use: True            # Enable boundary prediction
  tv_gamma: 1.0        # Gamma exponent for boundary-weighted smoothing
  loss_w: 0.5          # Weight for boundary BCE loss
```

### Hyperparameter Sweeps

Use Weights & Biases sweeps for systematic hyperparameter search.

#### Method 1: W&B Sweep Configuration (Bayesian Optimization)

See `jobs/sweep2.sh` for a complete example using Weights & Biases sweeps:
- Creates sweep YAML with Bayesian optimization
- Sweeps over learning rate, sample rate, and lr_decay
- Launches multiple agents on different GPUs
- Logs agents to separate files

#### Method 2: Grid Search with Bash Loop (Boundary Head Ablation)

See `jobs/sweep_boundary_grid.sh` for boundary head ablation:
- Grid search over boundary loss weight, TV gamma, and smoothness weight
- Sequential or parallel execution on multiple GPUs
- Organized logging per experiment

#### Monitor Sweep Results

View sweep results at: `https://wandb.ai/your_entity/your_project`

The W&B dashboard shows:
- Hyperparameter importance analysis
- Parallel coordinates plot
- Best runs ranked by metric
- Performance trends over time
- Automated hyperparameter optimization

### Distributed Training

For multi-GPU training, see the sweep scripts (`jobs/sweep2.sh`, `jobs/sweep_boundary_grid.sh`) which demonstrate:
- Launching agents on multiple GPUs using `CUDA_VISIBLE_DEVICES`
- Sequential vs parallel execution modes
- Background job management with nohup

## Evaluation

### Command-Line Evaluation

Evaluate a trained model on test split:

```bash
python -m FACT.eval \
    --cfg configs/your_config.yaml \
    --checkpoint checkpoints/best_model.net \
    --split test
```

Metrics computed:
- Frame-wise accuracy
- Segmental edit score
- Segmental F1-score (10%, 25%, 50% overlap thresholds)

### Inference and Visualization

For inference on validation or test videos with visualization:

See `visualization_simple.ipynb` for a complete workflow that includes:
- Loading trained model from submission directory
- Batch prediction on validation videos
- Computing metrics (accuracy, edit score, F1 scores)
- Creating horizontal bar visualizations
- Exporting predictions to text files

## Code Structure

```
FACT/
├── models/
│   ├── blocks.py          # FACT architecture, Block classes, boundary head
│   ├── basic.py           # Basic layers (MSTCN, attention, etc.)
│   ├── basic_hta.py       # Hierarchical temporal attention modules
│   └── loss.py            # Loss functions including boundary losses
├── utils/
│   ├── utils.py           # Utility functions
│   └── gen_config.py      # Configuration generation script
├── configs/
│   ├── utils.py           # Config utilities (update_from)
│   └── *.yaml             # Dataset-specific configurations
├── train.py               # Training script
├── eval.py                # Evaluation script
└── DATASET_SETUP_GUIDE.md # Dataset preparation guide
```

## Common Workflows

### Training a New Dataset

1. **Prepare dataset structure** (see DATASET_SETUP_GUIDE.md)
   ```bash
   dataset_name/
   ├── features/
   ├── groundTruth/
   ├── mapping.txt
   └── splits/
   ```

2. **Generate configuration**
   ```bash
   python FACT/utils/gen_config.py \
       --dataset_path /path/to/dataset \
       --dataset_name dataset_name \
       --base_config FACT/configs/breakfast.yaml \
       --feature_transpose \
       --output_config FACT/configs/dataset_name.yaml
   ```

3. **Train baseline model**
   ```bash
   python -m FACT.train \
       --cfg FACT/configs/dataset_name.yaml \
       --set split split1 \
            lr 3e-4 \
            batch_size 8 \
            sr 3
   ```

4. **Monitor training** in Weights & Biases dashboard

5. **Refine hyperparameters** with sweep (see Hyperparameter Sweeps section)

6. **Evaluate best model**
   ```bash
   python -m FACT.eval \
       --cfg FACT/configs/dataset_name.yaml \
       --checkpoint /path/to/best.net \
       --split test
   ```

### Reproducing Results with Boundary Head

Train with boundary head enabled:

```bash
python -m FACT.train \
    --cfg FACT/configs/sics3.yaml \
    --set Boundary.use true \
         Boundary.tv_gamma 2.0 \
         Boundary.loss_w 1.0 \
         Loss.sw 5.0 \
         split split1
```

### Ablation Study

See `jobs/sweep_boundary_grid.sh` for an example of systematic ablation over:
- Boundary head usage (enabled/disabled)
- Boundary loss weight
- TV gamma values
- Temporal smoothness weight

Adapt the grid search loop to test other components (architecture, frame branches, etc.)

### Fine-tuning from Pretrained Checkpoint

Resume training or fine-tune from an existing checkpoint:

```bash
python -m FACT.train \
    --cfg FACT/configs/sics3.yaml \
    --set aux.init_from /path/to/pretrained.net \
         aux.resume "" \
         lr 1e-4 \
         split split1
```

### Creating Submission Package

To package a trained model for submission:

1. Create new submission directory (e.g., `i3d_submission_3/`)
2. Copy FACT package with minimal configs (no yacs/scipy dependencies)
3. Copy trained checkpoint as `best.net`
4. Copy or adapt `model.py` with correct configuration
5. Zip without top-level folder

See existing `i3d_submission/` and `i3d_submission_3/` as templates.


