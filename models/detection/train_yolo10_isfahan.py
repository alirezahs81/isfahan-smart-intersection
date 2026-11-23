#!/usr/bin/env python3
"""
Train YOLOv10-ISF for Isfahan Traffic Detection
Supports: Synthetic data + Public datasets + Self-supervised pre-training
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv10 for Isfahan Traffic')
    parser.add_argument('--data', default='configs/isfahan_traffic.yaml', help='Dataset config')
    parser.add_argument('--model', default='yolov10n.pt', help='Base model')
    parser.add_argument('--epochs', type=int, default=150, help='Epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--device', default='0,1,2,3', help='GPU devices')
    parser.add_argument('--workers', type=int, default=16, help='Data workers')
    parser.add_argument('--project', default='runs/isfahan', help='Project directory')
    parser.add_argument('--name', default='yolov10n_isfahan_v1', help='Run name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--pretrained_mae', help='Path to MAE pre-trained encoder')
    args = parser.parse_args()
    
    # Load dataset config
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    print(f"🚀 Starting YOLOv10-ISF training")
    print(f"   Data: {args.data}")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch}")
    print(f"   Device: {args.device}")
    print(f"   Classes: {data_cfg['nc']} ({list(data_cfg['names'].values())})")
    
    # Initialize model
    model = YOLO(args.model)
    
    # Load MAE pre-trained encoder if provided
    if args.pretrained_mae and Path(args.pretrained_mae).exists():
        print(f"📦 Loading MAE pre-trained encoder from {args.pretrained_mae}")
        mae_ckpt = torch.load(args.pretrained_mae, map_location='cpu')
        # Transfer encoder weights (requires matching architecture)
        # model.model.load_state_dict(mae_ckpt['encoder'], strict=False)
        print("   Note: Manual weight transfer needed for MAE encoder")
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # LR Schedule
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        
        # Loss
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Augmentation (heavy for synthetic->real)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        
        # Validation
        val=True,
        save_json=True,  # COCO metrics
        save_hybrid=False,
        conf=0.001,
        iou=0.6,
        max_det=300,
        half=True,
        plots=True,
        save_period=10,
        
        # Reproducibility
        seed=42,
        deterministic=True,
    )
    
    print(f"✅ Training complete!")
    print(f"   Best mAP@0.5: {results.best_map50:.4f}")
    print(f"   Best mAP@0.5:0.95: {results.best_map:.4f}")
    print(f"   Weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == '__main__':
    main()