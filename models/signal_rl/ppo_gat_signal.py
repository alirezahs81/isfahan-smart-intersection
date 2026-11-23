#!/usr/bin/env python3
"""
PPO-GAT Signal Control for Isfahan Traffic Network
Graph Attention Network + PPO for multi-intersection adaptive signal control
Optimized for 5-intersection network on Chahar Bagh corridor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import yaml


class GATFeatureExtractor(BaseFeaturesExtractor):
    """
    Graph Attention Network for intersection network feature extraction
    Input: Graph-structured observation (nodes=intersections, edges=connectivity)
    Output: Fixed-size feature vector for PPO policy/value networks
    """
    
    def __init__(self, 
                 observation_space: gym.Space,
                 node_features: int = 19,
                 edge_features: int = 3,
                 hidden_dim: int = 128,
                 gat_heads: int = 4,
                 gat_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__(observation_space, features_dim=hidden_dim)
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # GAT Layers
        self.gat_layers = nn.ModuleList()
        for i in range(gat_layers):
            in_dim = hidden_dim * (gat_heads if i > 0 else 1)
            out_dim = hidden_dim
            heads = gat_heads if i < gat_layers - 1 else 1
            
            self.gat_layers.append(
                GATConv(
                    in_dim, out_dim, heads=heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                    concat=(i < gat_layers - 1),
                    add_self_loops=True
                )
            )
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * (gat_heads if i < gat_layers - 1 else 1))
            for i in range(gat_layers)
        ])
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, observations: Dict) -> torch.Tensor:
        """
        observations: Dict with keys:
        - 'node_features': [B, N, F_node] or [N, F_node]
        - 'edge_index': [2, E] or [B, 2, E]
        - 'edge_attr': [E, F_edge] or [B, E, F_edge]
        """
        # Handle batched and unbatched inputs
        if observations['node_features'].dim() == 2:
            # Single graph [N, F] -> add batch dim
            node_features = observations['node_features'].unsqueeze(0)  # [1, N, F]
            edge_index = observations['edge_index'].unsqueeze(0) if observations['edge_index'].dim() == 2 else observations['edge_index']
            edge_attr = observations['edge_attr'].unsqueeze(0) if observations['edge_attr'].dim() == 2 else observations['edge_attr']
            batched = False
        else:
            node_features = observations['node_features']  # [B, N, F]
            edge_index = observations['edge_index']
            edge_attr = observations['edge_attr']
            batched = True
        
        B, N, _ = node_features.shape
        
        # Embed nodes and edges
        x = self.node_embedding(node_features)  # [B, N, H]
        edge_attr_emb = self.edge_embedding(edge_attr)  # [B, E, H]
        
        # Process each graph in batch through GAT
        outputs = []
        for b in range(B):
            x_b = x[b]  # [N, H]
            ei_b = edge_index[b] if batched else edge_index.squeeze(0)
            ea_b = edge_attr_emb[b] if batched else edge_attr_emb.squeeze(0)
            
            # GAT layers
            for i, (gat, ln) in enumerate(zip(self.gat_layers, self.layer_norms)):
                x_b = gat(x_b, ei_b, ea_b)  # [N, H*heads] or [N, H]
                x_b = F.elu(x_b)
                x_b = ln(x_b)
            
            # Global pooling (mean over nodes)
            graph_feat = x_b.mean(dim=0)  # [H]
            outputs.append(graph_feat)
        
        # Stack batch
        out = torch.stack(outputs, dim=0)  # [B, H]
        
        # Final projection
        out = self.output_proj(out)  # [B, H]
        
        return out


class GATActorCriticPolicy(ActorCriticPolicy):
    """Custom PPO Policy with GAT Feature Extractor"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=GATFeatureExtractor,
                         features_extractor_kwargs={
                             'node_features': 19,
                             'edge_features': 3,
                             'hidden_dim': 256,
                             'gat_heads': 4,
                             'gat_layers': 3,
                         },
                         net_arch=dict(pi=[256, 128], vf=[256, 128]),
                         activation_fn=nn.ReLU)


class IsfahanSignalNetwork:
    """
    Isfahan 5-intersection network definition for Chahar Bagh corridor
    """
    
    INTERSECTIONS = {
        'chaharbag_azadi': {
            'index': 0,
            'name_fa': 'چهارباغ-آزادی',
            'phases': 5,
            'phase_defs': 'NB_SB_Straight,NB_SB_Left,EB_WB_Straight,EB_WB_Left,Pedestrian_All',
            'priority': 'main_artery',
            'coordinates': (0, 0),
            'approaches': ['north', 'south', 'east', 'west'],
        },
        'chaharbag_hezar_jerib': {
            'index': 1,
            'name_fa': 'چهارباغ-هزارجریب',
            'phases': 4,
            'phase_defs': 'Main_Straight,Main_Left,Side_Straight,Pedestrian',
            'priority': 'main_artery',
            'coordinates': (0, 800),  # 800m north
            'approaches': ['north', 'south', 'east', 'west'],
        },
        'chaharbag_jomhuri': {
            'index': 2,
            'name_fa': 'چهارباغ-جمهوری',
            'phases': 5,
            'phase_defs': 'NB_SB_Straight,NB_SB_Left,EB_WB_Straight,EB_WB_Left,Pedestrian_All',
            'priority': 'main_artery',
            'coordinates': (0, 1600),  # 1.6km north
            'approaches': ['north', 'south', 'east', 'west'],
        },
        'azadi_hezar_jerib': {
            'index': 3,
            'name_fa': 'آزادی-هزارجریب',
            'phases': 4,
            'phase_defs': 'Main_Straight,Main_Left,Side_Straight,Pedestrian',
            'priority': 'secondary',
            'coordinates': (-1200, 800),  # West of Chahar Bagh
            'approaches': ['north', 'south', 'east', 'west'],
        },
        'jomhuri_hezar_jerib': {
            'index': 4,
            'name_fa': 'جمهوری-هزارجریب',
            'phases': 4,
            'phase_defs': 'Main_Straight,Main_Left,Side_Straight,Pedestrian',
            'priority': 'secondary',
            'coordinates': (1200, 1600),  # East of Chahar Bagh
            'approaches': ['north', 'south', 'east', 'west'],
        },
    }
    
    # Graph connectivity (adjacency matrix)
    # Edges represent physical proximity for coordination
    ADJACENCY = [
        [0, 1, 0, 1, 0],  # chaharbag_azadi
        [1, 0, 1, 1, 0],  # chaharbag_hezar_jerib
        [0, 1, 0, 0, 1],  # chaharbag_jomhuri
        [1, 1, 0, 0, 0],  # azadi_hezar_jerib
        [0, 0, 1, 0, 0],  # jomhuri_hezar_jerib
    ]
    
    @classmethod
    def get_edge_index(cls) -> torch.Tensor:
        """Get edge_index for PyG [2, E]"""
        edges = []
        for i in range(5):
            for j in range(5):
                if cls.ADJACENCY[i][j]:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    @classmethod
    def get_edge_attr(cls) -> torch.Tensor:
        """Get edge attributes [E, 3]: [distance_km, correlation, spillback_risk]"""
        coords = [cls.INTERSECTIONS[k]['coordinates'] for k in sorted(cls.INTERSECTIONS.keys(), key=lambda x: cls.INTERSECTIONS[x]['index'])]
        edges = []
        
        for i in range(5):
            for j in range(5):
                if cls.ADJACENCY[i][j]:
                    # Distance
                    xi, yi = coords[i]
                    xj, yj = coords[j]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2) / 1000.0  # km
                    
                    # Correlation (traffic flow correlation)
                    corr = 0.7 if abs(i - j) == 1 else 0.4
                    
                    # Spillback risk
                    spillback = 0.8 if (i == 0 and j == 1) or (i == 1 and j == 0) else 0.3
                    
                    edges.append([dist, corr, spillback])
        
        return torch.tensor(edges, dtype=torch.float)
    
    @classmethod
    def get_phase_actions(cls) -> List[List[int]]:
        """Valid phase extension actions per intersection [0, 5, 10, 15, 20] seconds"""
        return [[0, 5, 10, 15, 20] for _ in range(5)]


class IsfahanSignalEnv(gym.Env):
    """
    Gymnasium Environment for Isfahan Signal Control
    Integrates with SUMO for realistic traffic simulation
    """
    
    def __init__(self, 
                 sumo_cfg: str = 'configs/sumo/isfahan_network.sumocfg',
                 gui: bool = False,
                 max_steps: int = 3600,  # 1 hour at 1-second steps
                 action_interval: int = 10):  # Decision every 10 seconds
        super().__init__()
        
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.max_steps = max_steps
        self.action_interval = action_interval
        
        # Network definition
        self.network = IsfahanSignalNetwork()
        self.num_intersections = 5
        
        # Observation space: Graph-structured
        # Node features per intersection (19 dims):
        # [queue_n, queue_s, queue_e, queue_w, 
        #  wait_n, wait_s, wait_e, wait_w,
        #  current_phase, phase_elapsed,
        #  adj_phase_n, adj_phase_s, adj_phase_e, adj_phase_w,
        #  ped_demand_n, ped_demand_s, ped_demand_e, ped_demand_w,
        #  emergency_present]
        self.observation_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(
                low=0, high=1, shape=(self.num_intersections, 19), dtype=np.float32
            ),
            'edge_index': gym.spaces.Box(
                low=0, high=4, shape=(2, 12), dtype=np.int32
            ),
            'edge_attr': gym.spaces.Box(
                low=0, high=1, shape=(12, 3), dtype=np.float32
            ),
        })
        
        # Action space: Phase extension per intersection [0, 5, 10, 15, 20] seconds
        self.action_space = gym.spaces.MultiDiscrete([5] * self.num_intersections)
        
        # SUMO connection (lazy initialization)
        self.sumo = None
        self.step_count = 0
        self.episode_reward = 0
        
        # Reward weights (tuned for Isfahan priorities)
        self.reward_weights = {
            'pressure': -1.0,           # Max Pressure (Varaiya)
            'wait_time': -0.5,          # Weighted waiting time
            'throughput': 0.3,          # Vehicle throughput
            'pedestrian_wait': -0.8,    # Pedestrian priority
            'emergency_priority': 5.0,  # Emergency vehicles (hospitals on Chahar Bagh)
            'phase_switch': -0.1,       # Smooth transitions
            'spillback': -2.0,          # Prevent queue spillback
            'equity': -0.3,             # Directional fairness
            'coordination': 0.5,        # Green wave coordination
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize SUMO
        if self.sumo is not None:
            self.sumo.close()
        
        from sumo_env import SumoEnvironment
        self.sumo = SumoEnvironment(
            net_file=self.sumo_cfg.replace('.sumocfg', '.net.xml'),
            route_file=self.sumo_cfg.replace('.sumocfg', '.rou.xml'),
            use_gui=self.gui,
            num_seconds=self.max_steps,
            delta_time=self.action_interval,
        )
        
        self.step_count = 0
        self.episode_reward = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute action: phase extensions for each intersection"""
        # Apply phase extensions
        phase_durations = self._compute_phase_durations(action)
        self.sumo.set_phase_durations(phase_durations)
        
        # Simulate
        self.sumo.simulate(self.action_interval)
        self.step_count += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        self.episode_reward += reward
        
        terminated = self.sumo.is_done()
        truncated = self.step_count >= (self.max_steps // self.action_interval)
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_phase_durations(self, action: np.ndarray) -> Dict:
        """Convert action indices to phase durations"""
        durations = {}
        
        for i, (name, info) in enumerate(self.network.INTERSECTIONS.items()):
            base_durations = self._get_base_phase_durations(info['phases'])
            extension = action[i] * 5  # 0, 5, 10, 15, 20 seconds
            
            # Apply extension to current phase
            current_phase = self.sumo.get_current_phase(name)
            if current_phase < len(base_durations):
                base_durations[current_phase] += extension
            
            durations[name] = base_durations
        
        return durations
    
    def _get_base_phase_durations(self, num_phases: int) -> List[int]:
        """Base phase durations from signal_phases.yaml"""
        if num_phases == 5:
            return [45, 15, 40, 12, 20]  # Standard 5-phase
        else:
            return [50, 15, 35, 15]  # Standard 4-phase
    
    def _get_observation(self) -> Dict:
        """Construct graph-structured observation"""
        node_features = np.zeros((self.num_intersections, 19), dtype=np.float32)
        
        for i, (name, info) in enumerate(self.network.INTERSECTIONS.items()):
            # Get traffic data from SUMO
            queues = self.sumo.get_queue_lengths(name)  # [N, S, E, W]
            waits = self.sumo.get_waiting_times(name)   # [N, S, E, W]
            phase = self.sumo.get_current_phase(name)
            phase_elapsed = self.sumo.get_phase_elapsed(name)
            adj_phases = self.sumo.get_adjacent_phases(name)
            ped_demands = self.sumo.get_pedestrian_demands(name)
            emergency = self.sumo.get_emergency_vehicles(name)
            
            # Normalize and pack
            node_features[i] = [
                queues[0] / 50.0, queues[1] / 50.0, queues[2] / 30.0, queues[3] / 30.0,
                waits[0] / 300.0, waits[1] / 300.0, waits[2] / 300.0, waits[3] / 300.0,
                phase / max(info['phases'], 1),
                phase_elapsed / 100.0,
                adj_phases[0] / 5.0, adj_phases[1] / 5.0, adj_phases[2] / 5.0, adj_phases[3] / 5.0,
                ped_demands[0] / 20.0, ped_demands[1] / 20.0, ped_demands[2] / 20.0, ped_demands[3] / 20.0,
                float(emergency > 0),
            ]
        
        # Clip to [0, 1]
        node_features = np.clip(node_features, 0, 1)
        
        return {
            'node_features': node_features,
            'edge_index': self.network.get_edge_index().numpy(),
            'edge_attr': self.network.get_edge_attr().numpy(),
        }
    
    def _compute_reward(self) -> float:
        """Compute multi-objective reward"""
        metrics = self.sumo.get_network_metrics()
        reward = 0.0
        
        for key, weight in self.reward_weights.items():
            if key in metrics:
                reward += weight * metrics[key]
        
        return reward
    
    def _get_info(self) -> Dict:
        """Additional info for logging"""
        metrics = self.sumo.get_network_metrics()
        return {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'metrics': metrics,
            'phase_durations': self.sumo.get_all_phase_durations(),
        }
    
    def close(self):
        if self.sumo is not None:
            self.sumo.close()
            self.sumo = None


def train_ppo_gat(config_path: str = 'configs/signal_rl.yaml'):
    """Train PPO-GAT for Isfahan signal control"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create vectorized environments
    n_envs = config.get('n_envs', 8)
    
    def make_env(rank):
        def _init():
            env = IsfahanSignalEnv(
                sumo_cfg=config['sumo_cfg'],
                gui=False,
                max_steps=config['max_steps'],
                action_interval=config['action_interval'],
            )
            return env
        return _init
    
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # Create model
    model = PPO(
        policy=GATActorCriticPolicy,
        env=env,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 256),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        tensorboard_log=config.get('tensorboard_log', './logs/ppo_gat/'),
        verbose=1,
        seed=42,
        device='cuda',
    )
    
    # Train
    total_timesteps = config.get('total_timesteps', 1_000_000)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save(config.get('model_save_path', 'models/signal_rl/weights/ppo_gat_isfahan'))
    
    print("✅ PPO-GAT training complete")
    return model


def export_policy(model: PPO, output_path: str):
    """Export policy to TorchScript for Triton/DeepStream"""
    policy = model.policy
    policy.eval()
    
    # Dummy input
    dummy_obs = {
        'node_features': torch.randn(1, 5, 19),
        'edge_index': torch.randint(0, 5, (2, 12)),
        'edge_attr': torch.rand(12, 3),
    }
    
    # Trace
    traced = torch.jit.trace(policy, dummy_obs)
    traced.save(output_path)
    
    print(f"✅ Policy exported to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/signal_rl.yaml')
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--n_envs', type=int, default=8)
    args = parser.parse_args()
    
    # Quick test
    env = IsfahanSignalEnv(gui=False)
    obs, _ = env.reset()
    print(f"Obs keys: {obs.keys()}")
    print(f"Node features: {obs['node_features'].shape}")
    print(f"Edge index: {obs['edge_index'].shape}")
    print(f"Edge attr: {obs['edge_attr'].shape}")
    
    # Test action
    action = np.array([0, 1, 2, 0, 1])  # Random extensions
    obs, reward, term, trunc, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    env.close()
    
    print("✅ Environment test passed")