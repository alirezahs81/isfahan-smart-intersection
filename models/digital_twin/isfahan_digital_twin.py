#!/usr/bin/env python3
"""
Isfahan Digital Twin - 3D Gaussian Splatting + SUMO Coupling
Real-time visualization & What-if analysis for ChaharBagh-Azadi intersection
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import time


@dataclass
class CameraPose:
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [roll, pitch, yaw] degrees
    fov: float
    resolution: Tuple[int, int]
    name: str
    name_fa: str


class GaussianSplattingRenderer:
    """
    3D Gaussian Splatting renderer for real-time novel view synthesis
    Optimized for intersection-scale scenes (50-100K Gaussians)
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = Path(model_path)
        self.gaussians = None
        self.sh_degree = 3
        self._load_model()
    
    def _load_model(self):
        """Load 3D Gaussian model from PLY or custom format"""
        # In production, load from COLMAP + Gaussian Splatting training output
        # For now, create placeholder
        print(f"Loading Gaussian model from {self.model_path}")
        # self.gaussians = torch.load(self.model_path, map_location=self.device)
        pass
    
    def render(self, camera_pose: CameraPose) -> Dict:
        """
        Render view from camera pose
        Returns: {'rgb': HxWx3, 'depth': HxW, 'alpha': HxW}
        """
        # Gaussian splatting rendering (simplified)
        # In production, use optimized CUDA kernels
        H, W = camera_pose.resolution
        
        # Placeholder render
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth = np.zeros((H, W), dtype=np.float32)
        alpha = np.zeros((H, W), dtype=np.float32)
        
        return {'rgb': rgb, 'depth': depth, 'alpha': alpha}
    
    def render_batch(self, camera_poses: List[CameraPose]) -> List[Dict]:
        """Batch render multiple views"""
        return [self.render(pose) for pose in camera_poses]


class SUMOInterface:
    """Interface to SUMO traffic simulation"""
    
    def __init__(self, net_file: str, route_file: str = None, gui: bool = False):
        self.net_file = net_file
        self.route_file = route_file
        self.gui = gui
        self.sumo = None
        self.vehicle_states = {}
        self.signal_states = {}
    
    def start(self):
        """Start SUMO simulation"""
        import traci
        cmd = ['sumo-gui' if self.gui else 'sumo', '-n', self.net_file]
        if self.route_file:
            cmd.extend(['-r', self.route_file])
        traci.start(cmd)
        self.sumo = traci
        print("✅ SUMO started")
    
    def step(self, seconds: float = 1.0):
        """Advance simulation"""
        if self.sumo:
            for _ in range(int(seconds / 0.1)):
                self.sumo.simulationStep()
            self._update_states()
    
    def _update_states(self):
        """Update internal state caches"""
        self.vehicle_states = {}
        for veh_id in self.sumo.vehicle.getIDList():
            self.vehicle_states[veh_id] = {
                'position': self.sumo.vehicle.getPosition(veh_id),
                'speed': self.sumo.vehicle.getSpeed(veh_id),
                'angle': self.sumo.vehicle.getAngle(veh_id),
                'type': self.sumo.vehicle.getTypeID(veh_id),
                'lane': self.sumo.vehicle.getLaneID(veh_id),
            }
        
        self.signal_states = {}
        for tls_id in self.sumo.trafficlight.getIDList():
            self.signal_states[tls_id] = {
                'phase': self.sumo.trafficlight.getPhase(tls_id),
                'phase_duration': self.sumo.trafficlight.getPhaseDuration(tls_id),
                'next_switch': self.sumo.trafficlight.getNextSwitch(tls_id),
            }
    
    def get_vehicle_states(self) -> Dict:
        return self.vehicle_states
    
    def get_signal_states(self) -> Dict:
        return self.signal_states
    
    def set_phase_durations(self, durations: Dict[str, List[int]]):
        """Set signal phase durations for intersections"""
        for tls_id, phase_durs in durations.items():
            if tls_id in self.sumo.trafficlight.getIDList():
                self.sumo.trafficlight.setPhaseDuration(tls_id, phase_durs)
    
    def run_scenario(self, signal_plan: Dict, duration: int = 3600) -> Dict:
        """Run what-if scenario with given signal plan"""
        # Reset simulation
        self.sumo.load(['-n', self.net_file, '-r', self.route_file])
        
        # Apply signal plan
        self.set_phase_durations(signal_plan)
        
        # Simulate
        metrics = {
            'total_wait': 0,
            'avg_wait': 0,
            'throughput': 0,
            'max_queue': 0,
            'emergency_served': 0,
            'ped_wait': 0,
        }
        
        for step in range(duration):
            self.step(1.0)
            # Collect metrics every 60 seconds
            if step % 60 == 0:
                step_metrics = self._compute_metrics()
                for k in metrics:
                    metrics[k] += step_metrics.get(k, 0)
        
        # Average
        for k in metrics:
            metrics[k] /= (duration / 60)
        
        return metrics
    
    def _compute_metrics(self) -> Dict:
        """Compute traffic metrics from current state"""
        total_wait = 0
        total_vehicles = 0
        max_queue = 0
        throughput = 0
        emergency_served = 0
        ped_wait = 0
        
        for veh_id, state in self.vehicle_states.items():
            wait_time = self.sumo.vehicle.getWaitingTime(veh_id)
            total_wait += wait_time
            total_vehicles += 1
            
            if 'emergency' in state['type'].lower():
                emergency_served += 1
        
        # Queue lengths per lane
        for lane_id in self.sumo.lane.getIDList():
            queue = self.sumo.lane.getLastStepHaltingNumber(lane_id)
            max_queue = max(max_queue, queue)
        
        return {
            'total_wait': total_wait,
            'avg_wait': total_wait / max(total_vehicles, 1),
            'throughput': throughput,
            'max_queue': max_queue,
            'emergency_served': emergency_served,
            'ped_wait': ped_wait,
        }
    
    def close(self):
        if self.sumo:
            self.sumo.close()


class IsfahanDigitalTwin:
    """
    Complete Digital Twin for ChaharBagh-Azadi Intersection
    Combines 3D Gaussian Splatting visualization with SUMO simulation
    """
    
    def __init__(self, 
                 gs_model_path: str = 'models/digital_twin/isfahan_gs.ply',
                 sumo_net: str = 'configs/sumo/isfahan_network.net.xml',
                 sumo_routes: str = 'configs/sumo/isfahan_routes.rou.xml',
                 camera_config: str = 'configs/camera_calibration.yaml'):
        
        self.gs_renderer = GaussianSplattingRenderer(gs_model_path)
        self.sumo = SUMOInterface(sumo_net, sumo_routes, gui=False)
        
        # Load camera poses
        self.cameras = self._load_camera_config(camera_config)
        
        # Persian labels for UI
        self.persian_labels = {
            'vehicle': 'خودرو',
            'pedestrian': 'پیاده‌رو',
            'queue_length': 'طول صف',
            'wait_time': 'زمان انتظار (ثانیه)',
            'signal_phase': 'فاز سیگنال',
            'anomaly_detected': '⚠️ ناهنجاری',
            'emergency_vehicle': '🚑 امدادرسانی',
            'throughput': 'عبور در ساعت',
            'congestion_level': 'سطح ترافیک',
        }
        
        # Sync rate
        self.sync_rate = 10  # Hz
        self.last_sync = 0
    
    def _load_camera_config(self, config_path: str) -> Dict[str, CameraPose]:
        """Load camera calibration and poses"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        cameras = {}
        for cam in config.get('cameras', []):
            cameras[cam['id']] = CameraPose(
                position=np.array(cam['location']),
                rotation=np.array(cam['rotation']),
                fov=cam['fov'],
                resolution=tuple(cam['resolution']),
                name=cam['id'],
                name_fa=cam['name_fa'],
            )
        return cameras
    
    def start(self):
        """Start SUMO simulation"""
        self.sumo.start()
    
    def sync(self):
        """Synchronize SUMO state to Gaussian scene"""
        self.sumo.step(1.0 / self.sync_rate)
        self.last_sync = time.time()
    
    def render_operator_view(self, camera_name: str = 'overview') -> np.ndarray:
        """Render real-time view for traffic control center operators"""
        if camera_name not in self.cameras:
            camera_name = list(self.cameras.keys())[0]
        
        camera = self.cameras[camera_name]
        
        # Render base scene
        render = self.gs_renderer.render(camera)
        rgb = render['rgb'].copy()
        
        # Overlay SUMO vehicles as 3D boxes
        self._draw_vehicles(rgb, camera)
        
        # Overlay signal status
        self._draw_signals(rgb, camera)
        
        # Overlay anomaly alerts
        self._draw_anomalies(rgb, camera)
        
        # Overlay UI elements (Persian)
        self._draw_ui_overlay(rgb, camera_name)
        
        return rgb
    
    def _draw_vehicles(self, rgb: np.ndarray, camera: CameraPose):
        """Draw 3D vehicle bounding boxes from SUMO"""
        # Project 3D vehicle positions to 2D image coordinates
        # This requires camera projection matrix
        pass
    
    def _draw_signals(self, rgb: np.ndarray, camera: CameraPose):
        """Draw signal phase status"""
        pass
    
    def _draw_anomalies(self, rgb: np.ndarray, camera: CameraPose):
        """Draw anomaly alerts on screen"""
        pass
    
    def _draw_ui_overlay(self, rgb: np.ndarray, camera_name: str):
        """Draw Persian UI overlay"""
        h, w = rgb.shape[:2]
        
        # Title bar
        cv2.rectangle(rgb, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(rgb, f"تقاطع {self.cameras[camera_name].name_fa}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Status indicators (example)
        cv2.putText(rgb, "وضعیت: 🟢 طبیعی", (w - 300, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def simulate_intervention(self, signal_plan: Dict, duration: int = 3600) -> Dict:
        """
        What-if analysis: test signal timing changes
        Returns comparison with baseline
        """
        print(f"🔮 Running what-if analysis for {duration}s...")
        
        # Run baseline (current plan)
        print("  Running baseline...")
        baseline = self.sumo.run_scenario(
            signal_plan=self._get_current_signal_plan(), 
            duration=duration
        )
        
        # Run proposed plan
        print("  Running proposed plan...")
        proposed = self.sumo.run_scenario(signal_plan, duration)
        
        # Compute improvements
        improvements = {
            'avg_wait_reduction_pct': (baseline['avg_wait'] - proposed['avg_wait']) / baseline['avg_wait'] * 100,
            'throughput_increase_pct': (proposed['throughput'] - baseline['throughput']) / baseline['throughput'] * 100,
            'queue_reduction_pct': (baseline['max_queue'] - proposed['max_queue']) / baseline['max_queue'] * 100,
            'emergency_improvement': proposed['emergency_served'] - baseline['emergency_served'],
            'ped_wait_reduction_pct': (baseline['ped_wait'] - proposed['ped_wait']) / baseline['ped_wait'] * 100,
        }
        
        recommendation = 'deploy' if improvements['avg_wait_reduction_pct'] > 10 else 'revise'
        
        return {
            'baseline': baseline,
            'proposed': proposed,
            'improvements': improvements,
            'recommendation': recommendation,
            'timestamp': time.time(),
        }
    
    def _get_current_signal_plan(self) -> Dict:
        """Extract current signal plan from SUMO"""
        # Read from configs/signal_phases.yaml
        import yaml
        with open('configs/signal_phases.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return {phase['name']: phase['duration'] for phase in config['phases']}
    
    def export_grafana_panels(self, output_path: str = 'digital_twin/grafana_panels.json'):
        """Export Grafana dashboard panels for Digital Twin"""
        panels = [
            # Live 3D View
            {
                'type': 'iframe',
                'title': 'نمای زنده ۳D تقاطع چهارباغ-آزادی',
                'url': '/digital_twin/live_view',
                'gridPos': {'x': 0, 'y': 0, 'w': 12, 'h': 8}
            },
            
            # KPIs
            {
                'type': 'stat',
                'title': 'طول صف شمال (وسله)',
                'targets': [{'expr': 'isfahan_queue_length{direction="north"}', 'legendFormat': ''}],
                'fieldConfig': {'defaults': {'unit': 'vehicles', 'thresholds': {'mode': 'absolute', 'steps': [{'color': 'green', 'value': 0}, {'color': 'yellow', 'value': 20}, {'color': 'red', 'value': 40}]}}},
                'gridPos': {'x': 0, 'y': 8, 'w': 6, 'h': 4}
            },
            {
                'type': 'stat',
                'title': 'طول صف جنوب (وسله)',
                'targets': [{'expr': 'isfahan_queue_length{direction="south"}', 'legendFormat': ''}],
                'fieldConfig': {'defaults': {'unit': 'vehicles'}},
                'gridPos': {'x': 6, 'y': 8, 'w': 6, 'h': 4}
            },
            {
                'type': 'stat',
                'title': 'طول صف شرق (وسله)',
                'targets': [{'expr': 'isfahan_queue_length{direction="east"}', 'legendFormat': ''}],
                'fieldConfig': {'defaults': {'unit': 'vehicles'}},
                'gridPos': {'x': 12, 'y': 8, 'w': 6, 'h': 4}
            },
            {
                'type': 'stat',
                'title': 'طول صف غرب (وسله)',
                'targets': [{'expr': 'isfahan_queue_length{direction="west"}', 'legendFormat': ''}],
                'fieldConfig': {'defaults': {'unit': 'vehicles'}},
                'gridPos': {'x': 18, 'y': 8, 'w': 6, 'h': 4}
            },
            
            # Signal Phase
            {
                'type': 'stat',
                'title': 'فاز فعال سیگنال',
                'targets': [{'expr': 'isfahan_signal_phase', 'legendFormat': ''}],
                'fieldConfig': {'defaults': {'mappings': [{'type': 'value', 'options': {'0': {'text': 'شمال-جنوب مستقیم', 'color': 'green'}, '1': {'text': 'شمال-جنوب چپ', 'color': 'yellow'}, '2': {'text': 'شرق-غرب مستقیم', 'color': 'blue'}, '3': {'text': 'شرق-غرب چپ', 'color': 'purple'}, '4': {'text': 'همه قرمز + پیاده‌رو', 'color': 'red'}}}]}},
                'gridPos': {'x': 0, 'y': 12, 'w': 6, 'h': 4}
            },
            
            # Anomaly Timeline
            {
                'type': 'timeseries',
                'title': 'ناهنجاری‌های فعال (۲۴ ساعت اخیر)',
                'targets': [
                    {'expr': 'isfahan_anomaly_by_type{type="red_light_violation"}', 'legendFormat': 'عبور از قرمز'},
                    {'expr': 'isfahan_anomaly_by_type{type="motorcycle_wrong_lane"}', 'legendFormat': 'موتور در خط اشتباه'},
                    {'expr': 'isfahan_anomaly_by_type{type="pedestrian_jaywalking"}', 'legendFormat': 'پیاده‌روی نامناسب'},
                    {'expr': 'isfahan_anomaly_by_type{type="wrong_way_driving"}', 'legendFormat': 'خلاف جهت'},
                ],
                'gridPos': {'x': 0, 'y': 16, 'w': 24, 'h': 8}
            },
            
            # What-if Analysis Button
            {
                'type': 'button',
                'title': 'تحلیل What-if',
                'gridPos': {'x': 0, 'y': 24, 'w': 6, 'h': 2},
                'options': {
                    'url': '/digital_twin/whatif',
                    'text': 'اجرای شبیه‌سازی برنامه جدید',
                    'icon': 'calculator'
                }
            },
        ]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(panels, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Grafana panels exported to {output_path}")
    
    def close(self):
        self.sumo.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs-model', default='models/digital_twin/isfahan_gs.ply')
    parser.add_argument('--sumo-net', default='configs/sumo/isfahan_network.net.xml')
    parser.add_argument('--sumo-routes', default='configs/sumo/isfahan_routes.rou.xml')
    parser.add_argument('--camera-config', default='configs/camera_calibration.yaml')
    parser.add_argument('--export-grafana', action='store_true')
    parser.add_argument('--whatif', action='store_true')
    args = parser.parse_args()
    
    twin = IsfahanDigitalTwin(
        gs_model_path=args.gs_model,
        sumo_net=args.sumo_net,
        sumo_routes=args.sumo_routes,
        camera_config=args.camera_config
    )
    
    if args.export_grafana:
        twin.export_grafana_panels()
    
    if args.whatif:
        # Example what-if
        new_plan = {
            'chaharbag_azadi': [50, 10, 35, 10, 25],  # Extended main artery green
        }
        result = twin.simulate_intervention(new_plan, duration=1800)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    twin.close()


if __name__ == '__main__':
    main()