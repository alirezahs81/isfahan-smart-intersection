#!/usr/bin/env python3
"""
Train OSNet-x1.0 ReID for Iranian Vehicle Re-identification
Fine-tuned on vehicle crops from Isfahan synthetic + public datasets
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import yaml
from pathlib import Path


class VehicleReIDDataset(torch.utils.data.Dataset):
    """Dataset for vehicle Re-ID training (triplet loss)"""
    
    def __init__(self, data_root: str, split: str = 'train', 
                 img_size: Tuple[int, int] = (256, 128)):
        self.data_root = Path(data_root) / split
        self.img_size = img_size
        
        # Collect tracklets: {track_id: [image_paths]}
        self.tracklets = {}
        for img_path in self.data_root.rglob('*.jpg'):
            # Expected naming: {track_id}_{frame}_{cam}.jpg
            parts = img_path.stem.split('_')
            if len(parts) >= 1:
                track_id = parts[0]
                if track_id not in self.tracklets:
                    self.tracklets[track_id] = []
                self.tracklets[track_id].append(str(img_path))
        
        # Filter tracklets with at least 2 images
        self.tracklets = {k: v for k, v in self.tracklets.items() if len(v) >= 2}
        self.track_ids = list(self.tracklets.keys())
        
        print(f"Loaded {len(self.tracklets)} tracklets with {sum(len(v) for v in self.tracklets.values())} images")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    
    def __len__(self):
        return len(self.track_ids)
    
    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        images = self.tracklets[track_id]
        
        # Sample 2 images for triplet (anchor, positive)
        img1_path, img2_path = torch.randperm(len(images))[:2].tolist()
        img1 = self._load_image(images[img1_path])
        img2 = self._load_image(images[img2_path])
        
        # Negative sample from different track
        neg_track_id = self.track_ids[torch.randint(0, len(self.track_ids), (1,)).item()]
        while neg_track_id == track_id:
            neg_track_id = self.track_ids[torch.randint(0, len(self.track_ids), (1,)).item()]
        neg_img_path = self.tracklets[neg_track_id][torch.randint(0, len(self.tracklets[neg_track_id]), (1,)).item()]
        img3 = self._load_image(neg_img_path)
        
        return img1, img2, img3, track_id
    
    def _load_image(self, path):
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return self.transform(img)


class OSNetReID(nn.Module):
    """OSNet-x1.0 for Vehicle Re-ID"""
    
    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            'osnet_x1_0',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        self.feature_dim = 512
        
        if num_classes > 0:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 512]
        features = F.normalize(features, p=2, dim=1)  # L2 normalize
        
        if self.classifier is not None:
            logits = self.classifier(features)
            return features, logits
        return features


class TripletLoss(nn.Module):
    """Batch-hard triplet loss with margin"""
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # anchor, positive, negative: [B, D]
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()


def train_reid(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_dataset = VehicleReIDDataset(args.data_root, 'train', (256, 128))
    val_dataset = VehicleReIDDataset(args.data_root, 'val', (256, 128))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # Model
    model = OSNetReID(num_classes=0).to(device)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss
    triplet_loss = TripletLoss(margin=args.margin).to(device)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for anchor, positive, negative, _ in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            feat_a = model(anchor)
            feat_p = model(positive)
            feat_n = model(negative)
            
            loss = triplet_loss(feat_a, feat_p, feat_n)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative, _ in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                feat_a = model(anchor)
                feat_p = model(positive)
                feat_n = model(negative)
                
                loss = triplet_loss(feat_a, feat_p, feat_n)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}')
        
        # Save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
            }, args.output_dir / 'osnet_x1_0_isfahan_best.pt')
            print(f'  ✅ Saved best model (val_loss={best_loss:.4f})')
        
        # Periodic save
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            }, args.output_dir / f'osnet_x1_0_isfahan_epoch_{epoch+1}.pt')
    
    print(f'✅ ReID training complete. Best val loss: {best_loss:.4f}')


def export_onnx(model_path: str, output_path: str):
    """Export trained ReID model to ONNX"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = OSNetReID(num_classes=0).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy input
    dummy = torch.randn(1, 3, 256, 128).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch'},
            'embedding': {0: 'batch'},
        }
    )
    
    print(f'✅ ONNX exported to {output_path}')


def export_tensorrt(onnx_path: str, engine_path: str):
    """Convert ONNX to TensorRT INT8 engine"""
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # INT8 calibration
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    engine_bytes = builder.build_serialized_network(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    
    print(f'✅ TensorRT engine saved to {engine_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/fused/reid', help='ReID dataset root')
    parser.add_argument('--output-dir', default='models/tracking/weights', help='Output directory')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--export-onnx', action='store_true')
    parser.add_argument('--export-tensorrt', action='store_true')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.export_onnx:
        export_onnx(
            Path(args.output_dir) / 'osnet_x1_0_isfahan_best.pt',
            Path(args.output_dir) / 'osnet_x1_0_isfahan.onnx'
        )
    elif args.export_tensorrt:
        export_tensorrt(
            Path(args.output_dir) / 'osnet_x1_0_isfahan.onnx',
            Path(args.output_dir) / 'osnet_x1_0_isfahan_int8.engine'
        )
    else:
        train_reid(args)