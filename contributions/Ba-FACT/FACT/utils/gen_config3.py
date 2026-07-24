import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
from yacs.config import CfgNode as CN

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from utils.utils import parse_label

def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML configuration files for FACT model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config and enable HTA+MSTCN with good default windows
  python utils/gen_config.py \\
    --dataset_path /path/to/dataset \\
    --dataset_name sics \\
    --base_config configs/breakfast.yaml \\
    --feature_transpose \\
    --output_config configs/sics.yaml \\
    --frame_branch m_hta \\
    --hta_heads 8 \\
    --hta_windows_Bi 16 64 256 \\
    --hta_windows_Bu 16 64 256 \\
    --hta_windows_BU 32 128 512

Tips:
  * Use '--base_config configs/gtea.yaml' for small datasets.
  * Use '--match_mode o2m' if videos often contain many repeated actions.
        """
    )

    # --- required dataset args ---
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset dir (must contain groundTruth/ and mapping.txt)')
    parser.add_argument('--dataset_name', type=str, default='new_dataset',
                        help='Dataset name (default: derived from dataset_path)')
    parser.add_argument('--base_config', type=str, required=True,
                        help='Path to base YAML config to extend')
    parser.add_argument('--output_config', type=str, default=None,
                        help='Output YAML path (default: config_{dataset}_{mode}.yaml)')
    parser.add_argument('--feature_transpose', action='store_true',
                        help='Set true if features are stored as (D x T) and need transposition')

    # --- training scale knobs ---
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size to write into YAML')
    parser.add_argument('--bg_class', type=int, default=0,
                        help='Background class label index')

    # --- token/matching knobs ---
    parser.add_argument('--token_ratio', type=float, default=1.5,
                        help='ntoken = int(max_segments * token_ratio) unless --ntoken given')
    parser.add_argument('--ntoken', type=int, default=None,
                        help='Override number of action tokens')
    parser.add_argument('--match_mode', choices=['o2o', 'o2m'], default='o2o',
                        help='Token-segment matching mode')

    # --- HTA/frame-branch knobs ---
    parser.add_argument('--frame_branch', choices=['m', 'm2', 'm_hta', 'm2_hta'],
                        default='m_hta', help='Frame branch type')
    parser.add_argument('--hta_heads', type=int, default=8,
                        help='Number of heads for HTARefine1D')
    # Allow per-block windows; if unspecified, fall back to solid defaults
    parser.add_argument('--hta_windows_Bi', type=int, nargs='+', default=None,
                        help='HTA windows (frames) for Bi block group, e.g., 16 64 256')
    parser.add_argument('--hta_windows_Bu', type=int, nargs='+', default=None,
                        help='HTA windows (frames) for Bu block group')
    parser.add_argument('--hta_windows_BU', type=int, nargs='+', default=None,
                        help='HTA windows (frames) for BU block group')
    # --- boundary head knobs ---
    parser.add_argument('--boundary', action='store_true',
                        help='Enable auxiliary boundary head + weighted TV.')
    parser.add_argument('--boundary_loss_w', type=float, default=0.5,
                        help='Weight for boundary BCE loss.')
    parser.add_argument('--boundary_tv_gamma', type=float, default=1.0,
                        help='Gamma for (1 - p_b)^gamma in weighted TV.')


    args = parser.parse_args()

    # Resolve dataset paths
    dataset_name = args.dataset_name or Path(args.dataset_path).name
    dataset_path = Path(args.dataset_path)

    feature_folder = dataset_path / 'features'
    label_folder   = dataset_path / 'groundTruth'
    mapping_file   = dataset_path / 'mapping.txt'
    splits_folder  = dataset_path / 'splits'

    if not label_folder.exists():
        raise FileNotFoundError(f"Label folder {label_folder} does not exist")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file {mapping_file} does not exist")

    print(f"Processing dataset: {dataset_name}")
    print(f"✓ Found label folder: {label_folder}")
    print(f"✓ Found mapping file: {mapping_file}")
    print("✓ Found feature folder:" if feature_folder.exists() else "⚠ Feature folder not found:", feature_folder)
    print("✓ Found splits folder:"  if splits_folder.exists()  else "⚠ Splits folder not found:",  splits_folder)

    # Analyze labels to compute stats for ntoken
    label_files = sorted(label_folder.glob('*.txt'))
    if not label_files:
        raise FileNotFoundError(f"No label files found in {label_folder}")

    lens = []
    for label_file in tqdm(label_files, desc="Analyzing labels"):
        with open(label_file, 'r') as f:
            labels = f.read().strip().split('\n')
        if args.match_mode == 'o2o':
            lens.append(len(parse_label(labels)))
        else:  # o2m
            lens.append(len(set(labels)))

    average_len = float(np.mean(lens))
    max_len     = int(np.max(lens))
    if args.ntoken is None:
        num_token = int(max_len * args.token_ratio)
    else:
        num_token = int(args.ntoken)

    print(f"Dataset Statistics:")
    print(f"  Avg #segments: {average_len:.2f}")
    print(f"  Max #segments: {max_len}")
    print(f"  Computed ntoken: {num_token}")

    # Load base config
    print(f"Loading base config: {args.base_config}")
    with open(args.base_config, 'r') as f:
        cfg = CN.load_cfg(f)

    # Allow adding new keys safely if the base cfg doesn't declare them
    try:
        cfg.set_new_allowed(True)
    except Exception:
        pass
    if hasattr(cfg, 'defrost'):
        cfg.defrost()

    # --- dataset-specific fields ---
    cfg.dataset           = dataset_name
    cfg.batch_size        = int(args.batch_size)
    cfg.feature_path      = str(feature_folder)
    cfg.groundTruth_path  = str(label_folder)
    cfg.split_path        = str(splits_folder)
    cfg.map_fname         = str(mapping_file)
    cfg.feature_transpose = bool(args.feature_transpose)
    cfg.bg_class          = int(args.bg_class)
    cfg.average_transcript_len = average_len

    # --- model knobs ---
    cfg.FACT.ntoken = int(num_token)
    cfg.Loss.match  = args.match_mode

    # --- frame branch selection + HTA knobs ---
    # defaults if user didn't pass per-block windows
    def _as_windows(x, fallback):
        return [int(v) for v in (x if x is not None else fallback)]

    default_Bi = [16, 64, 256]
    default_Bu = [16, 64, 256]
    default_BU = [32, 128, 512]

    win_Bi = _as_windows(args.hta_windows_Bi, default_Bi)
    win_Bu = _as_windows(args.hta_windows_Bu, default_Bu)
    win_BU = _as_windows(args.hta_windows_BU, default_BU)

    # Make sure nested nodes exist
    for node_name in ['Bi', 'Bu', 'BU']:
        if node_name not in cfg:
            cfg[node_name] = CN()

    # Set per-group frame branch
    cfg.Bi.f  = args.frame_branch
    cfg.Bu.f  = args.frame_branch
    cfg.BU.f  = args.frame_branch

    if 'hta' in args.frame_branch:
        # attach heads and windows per group
        cfg.Bi.f_nhead        = int(args.hta_heads)
        cfg.Bu.f_nhead        = int(args.hta_heads)
        cfg.BU.f_nhead        = int(args.hta_heads)

        cfg.Bi.f_hta_windows  = win_Bi
        cfg.Bu.f_hta_windows  = win_Bu
        cfg.BU.f_hta_windows  = win_BU
    
    # --- boundary head section ---
    if 'Boundary' not in cfg:
        cfg['Boundary'] = CN()
        cfg.Boundary.use = bool(args.boundary)
        cfg.Boundary.loss_w = float(args.boundary_loss_w)
        cfg.Boundary.tv_gamma = float(args.boundary_tv_gamma)


    # Freeze back if supported
    if hasattr(cfg, 'freeze'):
        cfg.freeze()

    # Save YAML
    output_config = args.output_config or f"config_{dataset_name}_{args.match_mode}.yaml"
    Path(output_config).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving config to: {output_config}")
    with open(output_config, 'w') as f:
        f.write(cfg.dump())

    print("✓ Config file generated successfully!")
    print("\nYou can still override at train time with `--set`, e.g.:")
    print("  --set Bi.f m2_hta Bi.f_hta_windows \"[8,32,128]\" ")

if __name__ == "__main__":
    main()
