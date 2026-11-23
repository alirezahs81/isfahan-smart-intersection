#!/usr/bin/env python3
"""
ST-GCN Anomaly Detection for Isfahan Traffic
Spatio-Temporal Graph Convolutional Network for trajectory-based anomaly classification
Supports 11 Iranian traffic anomaly classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import STConv
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path


class IsfahanSTGCN(nn.Module):
    """
    ST-GCN for traffic anomaly detection on intersection trajectories
    
    Input: [Batch, Timesteps, Nodes, Features] where:
    - Features: [x, y, vx, vy, ax, ay, class_embedding(6)] = 12 features
    - Nodes: Max 50 tracked objects per frame
    - Timesteps: 50 frames (2 seconds at 25 FPS)
    - Graph: Spatial (proximity + lane topology) + Temporal
    
    Output: [Batch, Nodes, 11] anomaly class logits + uncertainty
    """
    
    ANOMALY_CLASSES = [
        'normal',                    # 0
        'motorcycle_wrong_lane',     # 1
        'red_light_violation',       # 2
        'illegal_uturn',             # 3
        'stop_on_crosswalk',         # 4
        'wrong_way_driving',         # 5
        'pedestrian_jaywalking',     # 6
        'bus_lane_violation',        # 7
        'truck_in_car_lane',         # 8
        'sudden_lane_change',        # 9
        'speeding',                  # 10
    ]
    
    SEVERITY_WEIGHTS = {
        0: 0, 1: 3, 2: 10, 3: 5, 4: 4, 5: 10,
        6: 3, 7: 4, 8: 6, 9: 5, 10: 7
    }
    
    def __init__(self, 
                 num_nodes: int = 50,
                 in_channels: int = 12,
                 hidden_channels: int = 128,
                 num_classes: int = 11,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Class embedding for object type
        self.class_embedding = nn.Embedding(12, 6)  # 12 traffic classes
        
        # ST-GCN Layers
        self.st_convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels
            self.st_convs.append(
                STConv(in_ch, hidden_channels, hidden_channels, kernel_size)
            )
        
        # Temporal Attention
        self.use_attention = use_attention
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                hidden_channels, num_heads=8, dropout=dropout, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_channels)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Uncertainty Head (for active learning / OOD detection)
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Zone-specific adapters (for intersection zones)
        self.zone_adapters = nn.ModuleDict({
            'north_approach': nn.Linear(hidden_channels, hidden_channels),
            'south_approach': nn.Linear(hidden_channels, hidden_channels),
            'east_approach': nn.Linear(hidden_channels, hidden_channels),
            'west_approach': nn.Linear(hidden_channels, hidden_channels),
            'intersection_center': nn.Linear(hidden_channels, hidden_channels),
            'crosswalk': nn.Linear(hidden_channels, hidden_channels),
            'brt_lane': nn.Linear(hidden_channels, hidden_channels),
        })
    
    def forward(self, 
                x: torch.Tensor,           # [B, T, N, C]
                edge_index: torch.Tensor,  # [2, E]
                edge_weight: torch.Tensor, # [E]
                mask: Optional[torch.Tensor] = None,  # [B, T, N]
                zone_ids: Optional[torch.Tensor] = None  # [B, N] zone indices
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns: (logits, uncertainty) - [B, N, num_classes], [B, N]
        """
        B, T, N, C = x.shape
        
        # Project input features
        x = self.input_proj(x.view(-1, C)).view(B, T, N, -1)  # [B, T, N, H]
        
        # Apply ST-GCN layers
        for st_conv in self.st_convs:
            # STConv expects [N, C, T] format, we have [B, T, N, C]
            # Process each batch separately
            x_out = []
            for b in range(B):
                x_b = x[b].permute(1, 2, 0)  # [N, C, T]
                x_b = st_conv(x_b, edge_index, edge_weight)  # [N, C, T]
                x_b = F.gelu(x_b)
                x_out.append(x_b.permute(2, 0, 1))  # [T, N, C]
            x = torch.stack(x_out, dim=0)  # [B, T, N, H]
        
        # Temporal attention over timesteps
        if self.use_attention:
            # Reshape for attention: [B*N, T, H]
            x_flat = x.view(B * N, T, -1)
            
            if mask is not None:
                mask_flat = mask.view(B * N, T)
                attn_mask = ~mask_flat.bool()
            else:
                attn_mask = None
            
            # Self-attention
            x_attn, _ = self.temporal_attention(
                x_flat, x_flat, x_flat, key_padding_mask=attn_mask
            )
            x_attn = self.attention_norm(x_attn + x_flat)  # Residual
            
            # Use last timestep + attention-weighted context
            x_last = x_attn[:, -1]  # [B*N, H]
        else:
            # Simple last timestep
            x_last = x[:, -1].view(B * N, -1)  # [B*N, H]
        
        # Zone-specific adaptation
        if zone_ids is not None:
            zone_ids_flat = zone_ids.view(-1)  # [B*N]
            for zone_name, adapter in self.zone_adapters.items():
                zone_mask = (zone_ids_flat == self._get_zone_index(zone_name))
                if zone_mask.any():
                    x_last[zone_mask] = adapter(x_last[zone_mask])
        
        # Classification
        logits = self.classifier(x_last)  # [B*N, num_classes]
        uncertainty = self.uncertainty_head(x_last).squeeze(-1)  # [B*N]
        
        return logits.view(B, N, -1), uncertainty.view(B, N)
    
    def _get_zone_index(self, zone_name: str) -> int:
        zone_map = {
            'north_approach': 0, 'south_approach': 1,
            'east_approach': 2, 'west_approach': 3,
            'intersection_center': 4, 'crosswalk': 5, 'brt_lane': 6
        }
        return zone_map.get(zone_name, 0)
    
    def predict_with_confidence(self, 
                                x: torch.Tensor,
                                edge_index: torch.Tensor,
                                edge_weight: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                zone_ids: Optional[torch.Tensor] = None,
                                confidence_threshold: float = 0.7
                                ) -> Dict:
        """Predict with confidence scoring and uncertainty quantification"""
        self.eval()
        with torch.no_grad():
            logits, uncertainty = self.forward(x, edge_index, edge_weight, mask, zone_ids)
            
            probs = F.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            
            # High uncertainty flag
            high_uncertainty = uncertainty > uncertainty.quantile(0.9)
            # Low confidence flag
            low_confidence = max_probs < confidence_threshold
            # Needs human review
            needs_review = high_uncertainty | low_confidence
            
            # Severity scores
            severity = torch.tensor(
                [self.SEVERITY_WEIGHTS.get(p.item(), 0) for p in preds.flatten()],
                device=preds.device
            ).view_as(preds)
            
            return {
                'predictions': preds,           # [B, N]
                'probabilities': probs,         # [B, N, num_classes]
                'confidence': max_probs,        # [B, N]
                'uncertainty': uncertainty,     # [B, N]
                'needs_review': needs_review,   # [B, N]
                'severity': severity,           # [B, N]
                'anomaly_names': [self.ANOMALY_CLASSES[p] for p in preds.flatten().tolist()],
            }


def build_spatial_graph(num_nodes: int, 
                        positions: torch.Tensor,  # [N, 2] - x, y
                        lane_topology: Dict = None,
                        distance_threshold: float = 20.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build spatial graph for intersection tracking
    Edges based on: proximity + lane connectivity
    """
    edges = []
    edge_weights = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Euclidean distance
            dist = torch.norm(positions[i] - positions[j]).item()
            
            # Proximity edge
            if dist < distance_threshold:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
                weight = 1.0 / (1.0 + dist / 10.0)  # Inverse distance
                edge_weights.extend([weight, weight])
            
            # Lane topology edges (if available)
            if lane_topology and 'lane_adjacency' in lane_topology:
                if lane_topology['lane_adjacency'].get(i, set()) & lane_topology['lane_adjacency'].get(j, set()):
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_weights.extend([1.5, 1.5])  # Stronger connection
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    else:
        # Self-loops only
        edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
        edge_weight = torch.ones(num_nodes)
    
    return edge_index, edge_weight


def create_trajectory_features(tracks: List[Dict], 
                               max_timesteps: int = 50,
                               max_nodes: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert track data to ST-GCN input format
    tracks: List of track dicts with 'bbox_history', 'class_id', 'zone'
    Returns: (features [T, N, C], mask [T, N])
    """
    # Initialize
    features = torch.zeros(max_timesteps, max_nodes, 12)  # C=12
    mask = torch.zeros(max_timesteps, max_nodes, dtype=torch.bool)
    
    for n, track in enumerate(tracks[:max_nodes]):
        history = track.get('bbox_history', [])
        class_id = track.get('class_id', 0)
        zone = track.get('zone', 'unknown')
        
        for t, bbox in enumerate(history[-max_timesteps:]):
            if t >= max_timesteps:
                break
            
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Features: [cx, cy, vx, vy, ax, ay, class_emb(6)]
            features[t, n, 0] = cx
            features[t, n, 1] = cy
            
            # Velocity/acceleration from history
            if t > 0:
                prev_cx, prev_cy = features[t-1, n, 0], features[t-1, n, 1]
                features[t, n, 2] = cx - prev_cx  # vx
                features[t, n, 3] = cy - prev_cy  # vy
                
                if t > 1:
                    prev_vx, prev_vy = features[t-1, n, 2], features[t-1, n, 3]
                    features[t, n, 4] = features[t, n, 2] - prev_vx  # ax
                    features[t, n, 5] = features[t, n, 3] - prev_vy  # ay
            
            # Class one-hot (6 dims for main classes)
            class_onehot = torch.zeros(6)
            if class_id < 6:
                class_onehot[class_id] = 1
            features[t, n, 6:12] = class_onehot
            
            mask[t, n] = True
    
    return features.unsqueeze(0), mask.unsqueeze(0)  # Add batch dim [1, T, N, C]


def load_model(checkpoint_path: str, device: str = 'cuda') -> IsfahanSTGCN:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = IsfahanSTGCN(
        num_nodes=checkpoint.get('num_nodes', 50),
        in_channels=checkpoint.get('in_channels', 12),
        hidden_channels=checkpoint.get('hidden_channels', 128),
        num_classes=checkpoint.get('num_classes', 11),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def export_onnx(model: IsfahanSTGCN, 
                output_path: str,
                num_nodes: int = 50,
                timesteps: int = 50,
                device: str = 'cuda'):
    """Export model to ONNX for Triton/DeepStream"""
    model.eval()
    model.to(device)
    
    # Dummy inputs
    x = torch.randn(1, timesteps, num_nodes, 12).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 200)).to(device)
    edge_weight = torch.rand(200).to(device)
    mask = torch.ones(1, timesteps, num_nodes, dtype=torch.bool).to(device)
    zone_ids = torch.zeros(1, num_nodes, dtype=torch.long).to(device)
    
    torch.onnx.export(
        model,
        (x, edge_index, edge_weight, mask, zone_ids),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['trajectory', 'edge_index', 'edge_weight', 'mask', 'zone_ids'],
        output_names=['anomaly_logits', 'uncertainty'],
        dynamic_axes={
            'trajectory': {0: 'batch', 1: 'timesteps', 2: 'nodes'},
            'edge_index': {1: 'edges'},
            'edge_weight': {0: 'edges'},
            'mask': {0: 'batch', 1: 'timesteps', 2: 'nodes'},
            'zone_ids': {0: 'batch', 1: 'nodes'},
            'anomaly_logits': {0: 'batch', 1: 'nodes'},
            'uncertainty': {0: 'batch', 1: 'nodes'},
        }
    )
    
    print(f"✅ ONNX model exported to {output_path}")


if __name__ == '__main__':
    # Test model creation
    model = IsfahanSTGCN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    B, T, N = 2, 50, 30
    x = torch.randn(B, T, N, 12)
    edge_index = torch.randint(0, N, (2, 100))
    edge_weight = torch.rand(100)
    mask = torch.ones(B, T, N, dtype=torch.bool)
    zone_ids = torch.randint(0, 7, (B, N))
    
    logits, uncertainty = model(x, edge_index, edge_weight, mask, zone_ids)
    print(f"Output logits: {logits.shape}, uncertainty: {uncertainty.shape}")
    
    # Test prediction
    preds = model.predict_with_confidence(x, edge_index, edge_weight, mask, zone_ids)
    print(f"Predictions: {preds['predictions'].shape}")
    print(f"Sample: {preds['anomaly_names'][:5]}")