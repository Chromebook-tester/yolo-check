#!/usr/bin/env python3
"""
YOLOv5 Cable Check Training Script
노트북 케이블 단자 체크를 위한 YOLO 학습 스크립트

Usage:
    python train_cable_check.py --epochs 100 --batch-size 16
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add YOLOv5 root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from train import main as train_main, parse_opt as train_parse_opt


def parse_opt():
    """Parse command line arguments for cable check training."""
    parser = argparse.ArgumentParser(description='YOLOv5 Cable Check Training')
    
    # Essential arguments
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/laptop_cable_check.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train,val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='cable_check', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Cable check specific arguments
    parser.add_argument('--cable-classes', type=int, default=10, help='number of cable connection classes')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='minimum confidence threshold for detection')
    
    return parser.parse_args()


def setup_cable_check_training():
    """Setup specific configurations for cable check training."""
    print("🔌 Setting up Cable Check Training Configuration...")
    
    # Check if dataset exists
    data_path = ROOT / 'data' / 'laptop_cable_check.yaml'
    if not data_path.exists():
        print(f"❌ Dataset configuration file not found: {data_path}")
        sys.exit(1)
    
    # Load and validate dataset configuration
    with open(data_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"✅ Dataset: {data_config.get('path', 'Unknown')}")
    print(f"✅ Number of classes: {data_config.get('nc', 'Unknown')}")
    print(f"✅ Classes: {list(data_config.get('names', {}).values())}")
    
    # Check if training images exist
    dataset_root = Path(data_config['path'])
    train_path = dataset_root / data_config['train']
    val_path = dataset_root / data_config['val']
    
    if not train_path.exists():
        print(f"⚠️  Training images directory not found: {train_path}")
    else:
        train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
        print(f"📸 Training images found: {len(train_images)}")
    
    if not val_path.exists():
        print(f"⚠️  Validation images directory not found: {val_path}")
    else:
        val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
        print(f"📸 Validation images found: {len(val_images)}")
    
    return data_config


def main(opt):
    """Main training function for cable check."""
    print("🚀 Starting YOLOv5 Cable Check Training")
    print("=" * 50)
    
    # Setup cable check specific configuration
    data_config = setup_cable_check_training()
    
    # Update opt with cable check specific settings
    opt.data = str(ROOT / 'data' / 'laptop_cable_check.yaml')
    opt.name = 'cable_check_experiment'
    
    print(f"📊 Training Configuration:")
    print(f"   - Model: {opt.weights}")
    print(f"   - Dataset: {opt.data}")
    print(f"   - Epochs: {opt.epochs}")
    print(f"   - Batch Size: {opt.batch_size}")
    print(f"   - Image Size: {opt.imgsz}")
    print(f"   - Device: {opt.device}")
    print("=" * 50)
    
    # Start training using YOLOv5's main training function
    train_main(opt)
    
    print("🎉 Cable Check Training Completed!")


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
