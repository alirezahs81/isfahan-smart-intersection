#!/usr/bin/env python3
"""
Data Fusion Pipeline for Isfahan Traffic Detection
Merges: UA-DETRAC, BDD100K, NVIDIA AI City, Mapillary Vistas, Waymo Open
Unifies class taxonomy to 12 Isfahan classes
Outputs: YOLO format dataset + tracking annotations
"""

import argparse
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import Dataset, merge_samples
import numpy as np
from tqdm import tqdm


# Unified class taxonomy for Isfahan traffic
ISFAHAN_CLASS_MAP = {
    # UA-DETRAC classes
    'car': 'sedan_iran',
    'bus': 'bus_city',
    'van': 'van_transport',
    'truck': 'truck_heavy',
    
    # BDD100K classes
    'car': 'sedan_iran',
    'bus': 'bus_city',
    'truck': 'truck_heavy',
    'train': None,  # Not in Isfahan intersection
    'motorcycle': 'motorcycle_iran',
    'bicycle': 'bicycle',
    'person': 'pedestrian',
    'rider': 'pedestrian',
    'traffic light': 'traffic_light',
    'traffic sign': 'traffic_sign',
    
    # NVIDIA AI City
    'vehicle': 'sedan_iran',  # Will be refined by subtype
    'person': 'pedestrian',
    'bike': 'bicycle',
    
    # Mapillary Vistas
    'car': 'sedan_iran',
    'bus': 'bus_city',
    'truck': 'truck_heavy',
    'motorcycle': 'motorcycle_iran',
    'bicycle': 'bicycle',
    'person': 'pedestrian',
    'rider': 'pedestrian',
    'traffic light': 'traffic_light',
    'traffic sign': 'traffic_sign',
    'crosswalk': 'crosswalk',
    
    # Waymo
    'vehicle': 'sedan_iran',
    'pedestrian': 'pedestrian',
    'cyclist': 'bicycle',
    'sign': 'traffic_sign',
}

# Target classes (12 classes)
TARGET_CLASSES = [
    'sedan_iran', 'pickup_iran', 'motorcycle_iran', 'bus_brt',
    'bus_city', 'truck_heavy', 'van_transport', 'bicycle',
    'pedestrian', 'traffic_light', 'traffic_sign', 'crosswalk'
]

CLASS_TO_ID = {cls: i for i, cls in enumerate(TARGET_CLASSES)}


def load_uadetrac(data_dir: Path, split: str = 'train') -> Dataset:
    """Load UA-DETRAC dataset"""
    print(f"Loading UA-DETRAC {split}...")
    dataset = foz.load_zoo_dataset(
        "uadetrac", 
        split=split,
        dataset_dir=str(data_dir),
        label_types=["detections"],
        max_samples=None
    )
    return dataset


def load_bdd100k(data_dir: Path, split: str = 'train') -> Dataset:
    """Load BDD100K dataset"""
    print(f"Loading BDD100K {split}...")
    dataset = foz.load_zoo_dataset(
        "bdd100k",
        split=split,
        dataset_dir=str(data_dir),
        label_types=["detections", "tracklets"],
        max_samples=None
    )
    return dataset


def load_nvidia_aicity(data_dir: Path) -> Dataset:
    """Load NVIDIA AI City Challenge dataset"""
    print("Loading NVIDIA AI City...")
    # Custom loading for AI City format
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(data_dir),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="detections"
    )
    return dataset


def load_mapillary(data_dir: Path, split: str = 'train') -> Dataset:
    """Load Mapillary Vistas"""
    print(f"Loading Mapillary Vistas {split}...")
    dataset = foz.load_zoo_dataset(
        "mapillary-vistas",
        split=split,
        dataset_dir=str(data_dir),
        label_types=["detections", "segmentations"],
        max_samples=10000  # Limit for speed
    )
    return dataset


def unify_labels(sample) -> fo.Sample:
    """Map dataset-specific labels to unified Isfahan taxonomy"""
    if not hasattr(sample, 'ground_truth') or sample.ground_truth is None:
        return sample
    
    detections = sample.ground_truth.detections
    unified_detections = []
    
    for det in detections:
        original_label = det.label.lower()
        unified_label = ISFAHAN_CLASS_MAP.get(original_label)
        
        if unified_label and unified_label in CLASS_TO_ID:
            det.label = unified_label
            unified_detections.append(det)
        elif unified_label is None:
            # Skip unmapped classes
            pass
    
    if unified_detections:
        sample.ground_truth.detections = unified_detections
    else:
        sample.ground_truth = None
    
    return sample


def export_yolo_format(dataset: Dataset, output_dir: Path, split: str):
    """Export FiftyOne dataset to YOLO format"""
    print(f"Exporting {split} to YOLO format...")
    
    output_dir = Path(output_dir)
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    dataset.export(
        export_dir=str(output_dir / split),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=TARGET_CLASSES,
        split=split
    )
    
    # Create dataset.yaml
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': CLASS_TO_ID,
        'nc': len(TARGET_CLASSES),
    }
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)


def create_tracking_annotations(dataset: Dataset, output_dir: Path, split: str):
    """Create MOT-format tracking annotations"""
    print(f"Creating tracking annotations for {split}...")
    
    output_dir = Path(output_dir) / 'tracking'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mot_file = output_dir / f'{split}_tracks.txt'
    
    with open(mot_file, 'w') as f:
        for sample in tqdm(dataset, desc=f"Processing {split}"):
            if not sample.ground_truth or not sample.ground_truth.detections:
                continue
            
            frame_id = int(sample.filepath.split('/')[-1].split('.')[0])
            
            for det in sample.ground_truth.detections:
                if det.label not in CLASS_TO_ID:
                    continue
                
                # MOT format: frame, id, x, y, w, h, conf, class, visibility
                # Convert relative bbox to absolute
                x, y, w, h = det.bounding_box  # relative [0,1]
                class_id = CLASS_TO_ID[det.label]
                
                # Use tracking_id if available, else hash of label+position
                track_id = getattr(det, 'tracking_id', hash(f"{det.label}_{x}_{y}") % 100000)
                
                f.write(f"{frame_id},{track_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,{class_id},1\n")


def main():
    parser = argparse.ArgumentParser(description='Fuse public datasets for Isfahan traffic')
    parser.add_argument('--data-dir', default='data/public', help='Public datasets directory')
    parser.add_argument('--output-dir', default='data/fused', help='Output directory')
    parser.add_argument('--datasets', nargs='+', 
                       default=['uadetrac', 'bdd100k', 'mapillary'],
                       help='Datasets to fuse')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], help='Splits to process')
    parser.add_argument('--max-samples', type=int, help='Max samples per dataset')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_datasets = []
    
    for ds_name in args.datasets:
        for split in args.splits:
            if ds_name == 'uadetrac':
                dataset = load_uadetrac(data_dir, split)
            elif ds_name == 'bdd100k':
                dataset = load_bdd100k(data_dir, split)
            elif ds_name == 'mapillary':
                dataset = load_mapillary(data_dir, split)
            else:
                print(f"Unknown dataset: {ds_name}")
                continue
            
            # Apply class mapping
            dataset = dataset.map(unify_labels)
            
            # Filter empty samples
            dataset = dataset.filter_labels("ground_truth", lambda det: det.label in CLASS_TO_ID)
            
            # Limit samples
            if args.max_samples:
                dataset = dataset.take(args.max_samples)
            
            all_datasets.append(dataset)
            print(f"  {ds_name} {split}: {len(dataset)} samples")
    
    # Merge all datasets
    print("Merging datasets...")
    merged = merge_samples(all_datasets)
    print(f"Total merged samples: {len(merged)}")
    
    # Split train/val/test
    train_dataset = merged.match_tags("train")
    val_dataset = merged.match_tags("val")
    test_dataset = merged.match_tags("test") if "test" in args.splits else val_dataset.take(1000)
    
    # Export
    export_yolo_format(train_dataset, output_dir, 'train')
    export_yolo_format(val_dataset, output_dir, 'val')
    export_yolo_format(test_dataset, output_dir, 'test')
    
    # Tracking annotations
    create_tracking_annotations(train_dataset, output_dir, 'train')
    create_tracking_annotations(val_dataset, output_dir, 'val')
    
    # Statistics
    stats = {
        'total_samples': len(merged),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'classes': TARGET_CLASSES,
        'class_counts': {},
    }
    
    # Count class distribution
    for sample in merged:
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                stats['class_counts'][det.label] = stats['class_counts'].get(det.label, 0) + 1
    
    with open(output_dir / 'fusion_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("✅ Data fusion complete!")
    print(f"   Output: {output_dir}")
    print(f"   Train: {stats['train_samples']}, Val: {stats['val_samples']}, Test: {stats['test_samples']}")
    print(f"   Class distribution: {stats['class_counts']}")


if __name__ == '__main__':
    main()