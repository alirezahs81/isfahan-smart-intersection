#!/usr/bin/env python3
"""
Pipeline تولید داده سنتتیک برچسب‌دار برای ترافیک اصفهان
استفاده از CARLA با نقشه چهارباغ-آزادی
خروجی: تصاویر RGB + Depth + Semantic Segmentation + YOLO Labels + Tracking GT
"""

import carla
import random
import numpy as np
import cv2
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import threading


@dataclass
class CameraConfig:
    id: str
    name: str
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    fov: float
    resolution: Tuple[int, int]
    sensor_type: str = 'rgb'  # rgb, depth, semantic_segmentation


@dataclass
class VehicleSpec:
    blueprint: str
    class_name: str
    class_id: int
    color: Optional[str] = None


class IsfahanSyntheticDataGenerator:
    """
    تولیدکننده داده سنتتیک مخصوص تقاطع چهارباغ-آزادی اصفهان
    شامل: ترافیک واقعی، ناهنجاری‌های بومی، شرایط جوی متنوع
    """
    
    # کلاس‌های ایرانی (مطابق configs/anomalies_iran.yaml)
    IRAN_CLASSES = {
        'sedan_iran': 0,        # پژو ۴۰۵، پراید، تیبا، سمند
        'pickup_iran': 1,       # پژو پیکاپ، نیسان، فوتون
        'motorcycle_iran': 2,   # 뿔סר، سی‌جی، وچر
        'bus_brt': 3,           # اتوبوس BRT (خط اختصاصی)
        'bus_city': 4,          # اتوبوس شهری معمولی
        'truck_heavy': 5,       # کامیون سنگین
        'van_transport': 6,     # وانت حمل‌ونقل
        'bicycle': 7,
        'pedestrian': 8,
        'traffic_light': 9,
        'traffic_sign': 10,
        'crosswalk': 11,
    }
    
    # bleuprint‌های CARLA متناظر با خودروهای ایران
    IRANIAN_VEHICLES = {
        'sedan_iran': [
            'vehicle.peugeot.405', 'vehicle.saipa.pride', 'vehicle.ikco.tiba',
            'vehicle.peugeot.206', 'vehicle.renault.pars', 'vehicle.toyota.camry'
        ],
        'pickup_iran': [
            'vehicle.peugeot.pickup', 'vehicle.nissan.navara', 'vehicle.foton.tunland',
            'vehicle.chevrolet.colorado', 'vehicle.ford.ranger'
        ],
        'motorcycle_iran': [
            'vehicle.bajaj.pulsar', 'vehicle.honda.cg125', 'vehicle.yamaha.r15',
            'vehicle.kawasaki.ninja300', 'vehicle.suzuki.gsx150'
        ],
        'bus_brt': [
            'vehicle.mercedes.citaro', 'vehicle.man.lions_city', 'vehicle.volvo.7900',
            'vehicle.scania.citywide', 'vehicle.solaris.urbino'
        ],
        'bus_city': [
            'vehicle.mercedes.citaro', 'vehicle.man.lions_city', 'vehicle.volvo.b8rle',
            'vehicle.daf.sbf', 'vehicle.isuzu.gala'
        ],
        'truck_heavy': [
            'vehicle.volvo.fh16', 'vehicle.mercedes.actros', 'vehicle.scania.r500',
            'vehicle.man.tgs', 'vehicle.daf.xf', 'vehicle.iveco.stralis'
        ],
        'van_transport': [
            'vehicle.renault.kuango', 'vehicle.peugeot.boxer', 'vehicle.ford.transit',
            'vehicle.mercedes.sprinter', 'vehicle.vw.crafter', 'vehicle.iveco.daily'
        ],
    }
    
    # الگوهای ترافیک ساعتی اصفهان (نرمال‌شده)
    HOURLY_TRAFFIC_PATTERN = {
        0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.03, 5: 0.05,
        6: 0.15, 7: 0.45, 8: 0.95, 9: 0.85, 10: 0.65, 11: 0.60,
        12: 0.70, 13: 0.75, 14: 0.85, 15: 0.95, 16: 1.0, 17: 1.0,
        18: 0.95, 19: 0.80, 20: 0.60, 21: 0.40, 22: 0.25, 23: 0.10,
    }
    
    # توزیع انواع خودرو در اصفهان
    VEHICLE_DISTRIBUTION = {
        'sedan_iran': 0.42,
        'pickup_iran': 0.13,
        'motorcycle_iran': 0.22,
        'bus_brt': 0.02,
        'bus_city': 0.03,
        'truck_heavy': 0.02,
        'van_transport': 0.07,
        'bicycle': 0.03,
        'pedestrian': 0.06,
    }
    
    # ناهنجاری‌های قابل تزریق
    ANOMALY_TYPES = [
        'motorcycle_wrong_lane',
        'red_light_violation',
        'illegal_uturn',
        'stop_on_crosswalk',
        'wrong_way_driving',
        'pedestrian_jaywalking',
        'bus_lane_violation',
        'truck_in_car_lane',
        'sudden_lane_change',
        'speeding',
    ]
    
    def __init__(self, config_path: str = 'configs/intersection.yaml'):
        self.config = self._load_config(config_path)
        self.client = None
        self.world = None
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []
        self.cameras = []
        self.camera_queues = {}
        self.frame_counter = 0
        self.current_hour = 8  # شروع از اوج صبح
        self.anomaly_injection_rate = 0.1  # 10% فریم‌ها ناهنجاری دارند
        
        # Output directories
        self.output_dir = Path('data/synthetic')
        self._setup_output_dirs()
        
        # Camera configs from intersection.yaml
        self.camera_configs = self._parse_camera_configs()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'anomalies_injected': {a: 0 for a in self.ANOMALY_TYPES},
            'vehicle_counts': {k: 0 for k in self.VEHICLE_DISTRIBUTION.keys()},
        }
    
    def _load_config(self, config_path: str) -> dict:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_output_dirs(self):
        """ایجاد ساختار دایرکتوری خروجی"""
        dirs = [
            'images/train', 'images/val',
            'labels/train', 'labels/val',
            'depth/train', 'depth/val',
            'seg/train', 'seg/val',
            'tracking', 'annotations',
        ]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)
    
    def _parse_camera_configs(self) -> List[CameraConfig]:
        """استخراج تنظیمات دوربین از کانفیگ تقاطع"""
        configs = []
        for cam in self.config['intersection']['cameras']:
            configs.append(CameraConfig(
                id=cam['id'],
                name=cam['name_fa'],
                location=tuple(cam.get('calibration_location', cam.get('location', (0,0,8.5)))),
                rotation=tuple(cam.get('calibration_rotation', cam.get('rotation', (0,0,0)))),
                fov=cam['fov'],
                resolution=tuple(cam['resolution']),
                sensor_type='rgb'
            ))
            # Add depth and segmentation cameras at same positions
            for sensor_type in ['depth', 'semantic_segmentation']:
                configs.append(CameraConfig(
                    id=f"{cam['id']}_{sensor_type}",
                    name=f"{cam['name_fa']} {sensor_type}",
                    location=tuple(cam.get('calibration_location', cam.get('location', (0,0,8.5)))),
                    rotation=tuple(cam.get('calibration_rotation', cam.get('rotation', (0,0,0)))),
                    fov=cam['fov'],
                    resolution=tuple(cam['resolution']),
                    sensor_type=sensor_type
                ))
        return configs
    
    def connect_carla(self, host: str = 'localhost', port: int = 2000, timeout: float = 10.0):
        """اتصال به سرور CARLA"""
        print(f"🔌 Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        
        # Load Isfahan map
        print("🗺️ Loading Isfahan ChaharBagh-Azadi map...")
        self.world = self.client.load_world('Isfahan_ChaharBagh_Azadi')
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.04  # 25 FPS
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)
        
        print("✅ CARLA connected and map loaded")
        return self.world
    
    def spawn_traffic(self, base_vehicle_count: int = 120):
        """اسپان ترافیک بر اساس ساعت روز و توزیع اصفهان"""
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        if not spawn_points:
            print("⚠️ No spawn points found! Using manual spawn locations.")
            spawn_points = self._get_manual_spawn_points()
        
        # Calculate vehicle count for current hour
        hour_factor = self.HOURLY_TRAFFIC_PATTERN.get(self.current_hour, 0.5)
        vehicle_count = int(base_vehicle_count * hour_factor)
        
        print(f"🚗 Spawning {vehicle_count} vehicles (hour {self.current_hour}:00, factor {hour_factor:.2f})")
        
        # Clear existing
        self._clear_traffic()
        
        # Spawn vehicles
        spawned = 0
        attempts = 0
        max_attempts = vehicle_count * 3
        
        while spawned < vehicle_count and attempts < max_attempts:
            attempts += 1
            
            # Select vehicle class based on distribution
            vehicle_class = np.random.choice(
                list(self.VEHICLE_DISTRIBUTION.keys()),
                p=list(self.VEHICLE_DISTRIBUTION.values())
            )
            
            bp_name = random.choice(self.IRANIAN_VEHICLES[vehicle_class])
            bp = bp_lib.find(bp_name)
            
            # Set color for Iranian vehicles
            if bp.has_attribute('color'):
                colors = bp.get_attribute('color').recommended_values
                if vehicle_class == 'bus_brt':
                    bp.set_attribute('color', '255,0,0')  # Red for BRT
                elif vehicle_class == 'taxi':
                    bp.set_attribute('color', '255,255,0')  # Yellow
                elif colors:
                    bp.set_attribute('color', random.choice(colors))
            
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(bp, spawn_point)
            
            if vehicle:
                vehicle.set_autopilot(True)
                self.vehicles.append({
                    'actor': vehicle,
                    'class': vehicle_class,
                    'class_id': self.IRAN_CLASSES[vehicle_class],
                    'blueprint': bp_name,
                })
                self.stats['vehicle_counts'][vehicle_class] += 1
                spawned += 1
        
        # Spawn pedestrians
        self._spawn_pedestrians(int(vehicle_count * 0.3))
        
        print(f"✅ Spawned {len(self.vehicles)} vehicles, {len(self.walkers)} pedestrians")
    
    def _get_manual_spawn_points(self) -> List[carla.Transform]:
        """نقاط اسپان دستی برای روی знает تقاطع"""
        points = []
        half_iw = 15.0  # half intersection width
        approach_len = 200.0
        
        # North approach (southbound traffic)
        for lane in range(4):
            for dist in np.linspace(30, approach_len, 8):
                x = -half_iw + lane * 3.5 + 1.75
                y = dist
                points.append(carla.Transform(
                    carla.Location(x=x, y=y, z=0.5),
                    carla.Rotation(pitch=0, yaw=-90, roll=0)
                ))
        
        # South approach (northbound)
        for lane in range(4):
            for dist in np.linspace(30, approach_len, 8):
                x = half_iw - lane * 3.5 - 1.75
                y = -dist
                points.append(carla.Transform(
                    carla.Location(x=x, y=y, z=0.5),
                    carla.Rotation(pitch=0, yaw=90, roll=0)
                ))
        
        # East approach (westbound)
        for lane in range(3):
            for dist in np.linspace(30, approach_len, 6):
                x = dist
                y = half_iw - lane * 3.5 - 1.75
                points.append(carla.Transform(
                    carla.Location(x=x, y=y, z=0.5),
                    carla.Rotation(pitch=0, yaw=180, roll=0)
                ))
        
        # West approach (eastbound)
        for lane in range(3):
            for dist in np.linspace(30, approach_len, 6):
                x = -dist
                y = -half_iw + lane * 3.5 + 1.75
                points.append(carla.Transform(
                    carla.Location(x=x, y=y, z=0.5),
                    carla.Rotation(pitch=0, yaw=0, roll=0)
                ))
        
        return points
    
    def _spawn_pedestrians(self, count: int):
        """اسپان پیاده‌روها"""
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        
        for _ in range(count):
            bp = random.choice(walker_bps)
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            
            if spawn_point.location:
                walker = self.world.try_spawn_actor(bp, spawn_point)
                if walker:
                    controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                    if controller:
                        controller.start()
                        # Random destination
                        dest = self.world.get_random_location_from_navigation()
                        if dest:
                            controller.go_to_location(dest)
                        self.walkers.append(walker)
                        self.walker_controllers.append(controller)
                        self.stats['vehicle_counts']['pedestrian'] += 1
    
    def setup_cameras(self):
        """راه‌اندازی دوربین‌ها"""
        bp_lib = self.world.get_blueprint_library()
        
        for cam_config in self.camera_configs:
            if cam_config.sensor_type == 'rgb':
                bp = bp_lib.find('sensor.camera.rgb')
            elif cam_config.sensor_type == 'depth':
                bp = bp_lib.find('sensor.camera.depth')
            elif cam_config.sensor_type == 'semantic_segmentation':
                bp = bp_lib.find('sensor.camera.semantic_segmentation')
            else:
                continue
            
            bp.set_attribute('image_size_x', str(cam_config.resolution[0]))
            bp.set_attribute('image_size_y', str(cam_config.resolution[1]))
            bp.set_attribute('fov', str(cam_config.fov))
            bp.set_attribute('sensor_tick', '0.04')  # 25 FPS
            
            transform = carla.Transform(
                carla.Location(*cam_config.location),
                carla.Rotation(*cam_config.rotation)
            )
            
            camera = self.world.spawn_actor(bp, transform)
            
            # Create queue for this camera
            queue = queue.Queue()
            camera.listen(queue.put)
            self.camera_queues[cam_config.id] = queue
            self.cameras.append({
                'actor': camera,
                'config': cam_config,
                'queue': queue,
            })
        
        print(f"📷 Setup {len(self.cameras)} cameras")
    
    def inject_anomalies(self, intensity: float = 0.15):
        """تزریق ناهنجاری‌های بومی اصفهان"""
        if not self.vehicles:
            return
        
        num_anomalies = int(len(self.vehicles) * intensity)
        anomaly_types = random.choices(self.ANOMALY_TYPES, k=num_anomalies)
        
        for anomaly_type in anomaly_types:
            self._inject_single_anomaly(anomaly_type)
            self.stats['anomalies_injected'][anomaly_type] += 1
    
    def _inject_single_anomaly(self, anomaly_type: str):
        """تزریق یک ناهنجاری خاص"""
        if anomaly_type == 'motorcycle_wrong_lane':
            self._inject_motorcycle_wrong_lane()
        elif anomaly_type == 'red_light_violation':
            self._inject_red_light_violation()
        elif anomaly_type == 'illegal_uturn':
            self._inject_illegal_uturn()
        elif anomaly_type == 'stop_on_crosswalk':
            self._inject_stop_on_crosswalk()
        elif anomaly_type == 'wrong_way_driving':
            self._inject_wrong_way_driving()
        elif anomaly_type == 'pedestrian_jaywalking':
            self._inject_pedestrian_jaywalking()
        elif anomaly_type == 'bus_lane_violation':
            self._inject_bus_lane_violation()
        elif anomaly_type == 'truck_in_car_lane':
            self._inject_truck_in_car_lane()
        elif anomaly_type == 'sudden_lane_change':
            self._inject_sudden_lane_change()
        elif anomaly_type == 'speeding':
            self._inject_speeding()
    
    def _inject_motorcycle_wrong_lane(self):
        """موتور در خط BRT یا چپidelان"""
        motorcycles = [v for v in self.vehicles if v['class'] == 'motorcycle_iran']
        if not motorcycles:
            return
        m = random.choice(motorcycles)
        m['actor'].set_autopilot(False)
        # Force into BRT lane (lane 4 on Chahar Bagh)
        loc = m['actor'].get_location()
        # Move to BRT lane position
        if abs(loc.x) < 5:  # On Chahar Bagh
            new_y = loc.y
            new_x = -12.0 if loc.y > 0 else 12.0  # BRT lane offset
            m['actor'].set_location(carla.Location(new_x, new_y, loc.z))
    
    def _inject_red_light_violation(self):
        """عبور از قرمز - غیرفعال کردن اتوپایلوت و اجبار به عبور"""
        vehicles = [v for v in self.vehicles if v['class'] in ['sedan_iran', 'pickup_iran', 'van_transport']]
        if not vehicles:
            return
        v = random.choice(vehicles)
        v['actor'].set_autopilot(False)
        # Give velocity through intersection
        v['actor'].enable_constant_velocity(carla.Vector3D(0, -8, 0) if v['actor'].get_location().y > 0 else carla.Vector3D(0, 8, 0))
    
    def _inject_illegal_uturn(self):
        """دور زدن غیرمجاز"""
        vehicles = [v for v in self.vehicles if v['class'] in ['sedan_iran', 'pickup_iran']]
        if not vehicles:
            return
        v = random.choice(vehicles)
        v['actor'].set_autopilot(False)
        # Sharp turn
        v['actor'].enable_constant_velocity(carla.Vector3D(5, 5, 0))
    
    def _inject_stop_on_crosswalk(self):
        """توقف روی پیاده‌رو"""
        vehicles = [v for v in self.vehicles if v['class'] in ['sedan_iran', 'pickup_iran', 'van_transport']]
        if not vehicles:
            return
        v = random.choice(vehicles)
        v['actor'].set_autopilot(False)
        v['actor'].set_target_velocity(carla.Vector3D(0, 0, 0))
        # Position on crosswalk
        loc = v['actor'].get_location()
        if abs(loc.x) < 15 and abs(loc.y) < 15:
            v['actor'].set_location(carla.Location(loc.x, loc.y, loc.z))
    
    def _inject_wrong_way_driving(self):
        """رانندگی خلاف جهت"""
        vehicles = [v for v in self.vehicles if v['class'] in ['sedan_iran', 'pickup_iran', 'motorcycle_iran']]
        if not vehicles:
            return
        v = random.choice(vehicles)
        v['actor'].set_autopilot(False)
        # Reverse direction
        vel = v['actor'].get_velocity()
        v['actor'].enable_constant_velocity(carla.Vector3D(-vel.x, -vel.y, 0))
    
    def _inject_pedestrian_jaywalking(self):
        """پیاده‌روی نامناسب"""
        if not self.walkers:
            return
        w = random.choice(self.walkers)
        # Move to random location not on crosswalk
        loc = w.get_location()
        offset = random.uniform(-10, 10)
        w.set_location(carla.Location(loc.x + offset, loc.y, loc.z))
    
    def _inject_bus_lane_violation(self):
        """خودرو سواری در خط BRT"""
        cars = [v for v in self.vehicles if v['class'] in ['sedan_iran', 'pickup_iran', 'van_transport']]
        if not cars:
            return
        v = random.choice(cars)
        v['actor'].set_autopilot(False)
        loc = v['actor'].get_location()
        if abs(loc.x) < 5:  # On Chahar Bagh
            new_x = -12.0 if loc.y > 0 else 12.0
            v['actor'].set_location(carla.Location(new_x, loc.y, loc.z))
    
    def _inject_truck_in_car_lane(self):
        """کامیون در خط سواری"""
        trucks = [v for v in self.vehicles if v['class'] == 'truck_heavy']
        if not trucks:
            return
        v = random.choice(trucks)
        v['actor'].set_autopilot(False)
        # Move to car lane (not BRT)
        loc = v['actor'].get_location()
        if abs(loc.x) < 5:
            new_x = -4.0 if loc.y > 0 else 4.0
            v['actor'].set_location(carla.Location(new_x, loc.y, loc.z))
    
    def _inject_sudden_lane_change(self):
        """تغییر خط ناگهانی"""
        vehicles = [v for v in self.vehicles if v['class'] in ['sedan_iran', 'pickup_iran', 'motorcycle_iran']]
        if not vehicles:
            return
        v = random.choice(vehicles)
        v['actor'].set_autopilot(False)
        # Sudden lateral movement
        v['actor'].enable_constant_velocity(carla.Vector3D(8, 5, 0))
    
    def _inject_speeding(self):
        """سرعت بیش از حد"""
        vehicles = [v for v in self.vehicles if v['class'] in ['sedan_iran', 'pickup_iran', 'motorcycle_iran']]
        if not vehicles:
            return
        v = random.choice(vehicles)
        v['actor'].set_autopilot(False)
        # High speed
        v['actor'].enable_constant_velocity(carla.Vector3D(0, -20, 0))
    
    def capture_frame(self, split: str = 'train') -> Dict:
        """ضبط یک فریم از تمام دوربین‌ها + تولید برچسب‌ها"""
        self.world.tick()
        self.frame_counter += 1
        
        frame_data = {
            'frame_id': self.frame_counter,
            'timestamp': time.time(),
            'hour': self.current_hour,
            'cameras': {},
            'ground_truth': {
                'vehicles': [],
                'pedestrians': [],
                'anomalies': [],
                'signal_phase': self._get_current_signal_phase(),
            }
        }
        
        # Collect camera data
        for cam in self.cameras:
            try:
                image = cam['queue'].get(timeout=1.0)
                cam_data = self._process_camera_image(image, cam['config'])
                frame_data['cameras'][cam['config'].id] = cam_data
            except queue.Empty:
                print(f"⚠️ Camera {cam['config'].id} timeout")
        
        # Collect ground truth
        frame_data['ground_truth']['vehicles'] = self._get_vehicle_ground_truth()
        frame_data['ground_truth']['pedestrians'] = self._get_pedestrian_ground_truth()
        frame_data['ground_truth']['anomalies'] = self._detect_anomalies_in_frame()
        
        # Save to disk
        self._save_frame(frame_data, split)
        
        self.stats['frames_captured'] += 1
        return frame_data
    
    def _process_camera_image(self, image: carla.Image, config: CameraConfig) -> Dict:
        """پردازش تصویر CARLA"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((config.resolution[1], config.resolution[0], 4))  # RGBA
        
        if config.sensor_type == 'semantic_segmentation':
            # CARLA semantic segmentation: R channel = class id
            seg = array[:, :, 2]  # Red channel
            return {
                'type': 'segmentation',
                'data': seg,
                'classes': self._get_carla_semantic_classes(),
            }
        elif config.sensor_type == 'depth':
            # Depth: normalized to [0, 1] in RGB channels
            depth = (array[:, :, 0] + array[:, :, 1] * 256 + array[:, :, 2] * 256 * 256) / (256**3 - 1)
            depth_meters = depth * 1000  # Convert to meters (max 1000m)
            return {
                'type': 'depth',
                'data': depth_meters.astype(np.float32),
            }
        else:
            # RGB
            rgb = array[:, :, :3][:, :, ::-1]  # BGRA to RGB
            return {
                'type': 'rgb',
                'data': rgb,
            }
    
    def _get_vehicle_ground_truth(self) -> List[Dict]:
        """استخراج Ground Truth برای خودروها (2D BBox + 3D + Class + Velocity)"""
        gt = []
        for v in self.vehicles:
            actor = v['actor']
            if not actor.is_alive:
                continue
            
            transform = actor.get_transform()
            location = transform.location
            rotation = transform.rotation
            velocity = actor.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            
            # 3D Bounding box
            bbox = actor.bounding_box
            extent = bbox.extent
            
            gt.append({
                'id': actor.id,
                'class': v['class'],
                'class_id': v['class_id'],
                'blueprint': v['blueprint'],
                'location': {'x': location.x, 'y': location.y, 'z': location.z},
                'rotation': {'pitch': rotation.pitch, 'yaw': rotation.yaw, 'roll': rotation.roll},
                'extent': {'x': extent.x, 'y': extent.y, 'z': extent.z},
                'velocity': {'x': velocity.x, 'y': velocity.y, 'z': velocity.z},
                'speed_kmh': speed,
                'lane_id': self._get_lane_id(location),
            })
        return gt
    
    def _get_pedestrian_ground_truth(self) -> List[Dict]:
        gt = []
        for w in self.walkers:
            if not w.is_alive:
                continue
            transform = w.get_transform()
            loc = transform.location
            vel = w.get_velocity()
            gt.append({
                'id': w.id,
                'class': 'pedestrian',
                'class_id': self.IRAN_CLASSES['pedestrian'],
                'location': {'x': loc.x, 'y': loc.y, 'z': loc.z},
                'velocity': {'x': vel.x, 'y': vel.y, 'z': vel.z},
                'speed_kmh': 3.6 * math.sqrt(vel.x**2 + vel.y**2),
            })
        return gt
    
    def _detect_anomalies_in_frame(self) -> List[Dict]:
        """تشخیص ناهنجاری‌های فعال در فریم جاری"""
        anomalies = []
        # This would be populated by the anomaly injection logic
        # For synthetic data, we know what we injected
        return anomalies
    
    def _get_current_signal_phase(self) -> Dict:
        """فاز فعلی سیگنال (شبیه‌سازی)"""
        # Simplified: cycle through phases
        cycle_time = 132  # Total cycle time in seconds
        elapsed = (self.frame_counter * 0.04) % cycle_time
        
        phases = [
            (0, 45, 'NB_SB_Straight'),
            (45, 15, 'NB_SB_Left'),
            (60, 40, 'EB_WB_Straight'),
            (100, 12, 'EB_WB_Left'),
            (112, 20, 'All_Red_Pedestrian'),
        ]
        
        for start, dur, name in phases:
            if start <= elapsed < start + dur:
                return {'phase_id': phases.index((start, dur, name)), 'name': name, 'elapsed': elapsed - start}
        
        return {'phase_id': 0, 'name': 'NB_SB_Straight', 'elapsed': 0}
    
    def _get_lane_id(self, location: carla.Location) -> int:
        """تعیین خط بر اساس مختصات"""
        x, y = location.x, location.y
        half_iw = 15.0
        
        if abs(x) < half_iw and abs(y) < half_iw:
            return 0  # Inside intersection
        
        if abs(x) < 3.0:  # Chahar Bagh lanes
            if y > 0:  # North approach
                if x < -10: return 4  # BRT
                elif x < -6.5: return 3
                elif x < -3: return 2
                else: return 1
            else:  # South approach
                if x > 10: return 4  # BRT
                elif x > 6.5: return 3
                elif x > 3: return 2
                else: return 1
        elif abs(y) < 3.0:  # Azadi lanes
            if x > 0:  # East approach
                if y > 6.5: return 3
                elif y > 3: return 2
                else: return 1
            else:  # West approach
                if y < -6.5: return 3
                elif y < -3: return 2
                else: return 1
        return -1
    
    def _get_carla_semantic_classes(self) -> Dict[int, str]:
        """Mapping کلاس‌های CARLA semantic segmentation"""
        return {
            0: 'none', 1: 'buildings', 2: 'fences', 3: 'other',
            4: 'pedestrians', 5: 'poles', 6: 'road_lines', 7: 'roads',
            8: 'sidewalks', 9: 'vegetation', 10: 'vehicles', 11: 'walls',
            12: 'traffic_signs', 13: 'sky', 14: 'ground', 15: 'bridge',
            16: 'rail_track', 17: 'guard_rail', 18: 'traffic_light',
            19: 'static', 20: 'dynamic', 21: 'water', 22: 'terrain'
        }
    
    def _save_frame(self, frame_data: Dict, split: str):
        """ذخیره فریم و برچسب‌ها به دیسک"""
        frame_id = frame_data['frame_id']
        
        for cam_id, cam_data in frame_data['cameras'].items():
            base_name = f"{split}_{frame_id:06d}_{cam_id}"
            
            if cam_data['type'] == 'rgb':
                # Save image
                img_path = self.output_dir / 'images' / split / f"{base_name}.jpg"
                cv2.imwrite(str(img_path), cam_data['data'][:, :, ::-1])  # RGB to BGR
                
                # Save YOLO labels
                self._save_yolo_labels(frame_data, cam_id, base_name, split)
                
            elif cam_data['type'] == 'depth':
                depth_path = self.output_dir / 'depth' / split / f"{base_name}.npy"
                np.save(str(depth_path), cam_data['data'])
                
            elif cam_data['type'] == 'segmentation':
                seg_path = self.output_dir / 'seg' / split / f"{base_name}.png"
                cv2.imwrite(str(seg_path), cam_data['data'].astype(np.uint8))
        
        # Save tracking annotations (per frame)
        self._save_tracking_annotation(frame_data, split)
        
        # Save full annotation JSON periodically
        if frame_id % 100 == 0:
            ann_path = self.output_dir / 'annotations' / f"{split}_{frame_id:06d}.json"
            with open(ann_path, 'w') as f:
                # Convert numpy arrays to lists
                serializable = self._make_serializable(frame_data)
                json.dump(serializable, f, indent=2)
    
    def _save_yolo_labels(self, frame_data: Dict, cam_id: str, base_name: str, split: str):
        """تولید برچسب YOLO از Ground Truth با پروژکشن به دوربین"""
        # اینجا باید پروژکشن 3D به 2D انجام شود
        # برای ساده‌سازی، از bounding box‌های 2D تقدیری استفاده می‌کنیم
        
        cam_config = next(c for c in self.camera_configs if c.id == cam_id)
        w, h = cam_config.resolution
        
        label_path = self.output_dir / 'labels' / split / f"{base_name}.txt"
        
        with open(label_path, 'w') as f:
            for v in frame_data['ground_truth']['vehicles']:
                # Project 3D to 2D (simplified)
                bbox_2d = self._project_3d_to_2d(v, cam_config)
                if bbox_2d:
                    x_center, y_center, width, height = bbox_2d
                    # Normalize
                    x_center /= w
                    y_center /= h
                    width /= w
                    height /= h
                    
                    if 0 < x_center < 1 and 0 < y_center < 1 and width > 0 and height > 0:
                        f.write(f"{v['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            for p in frame_data['ground_truth']['pedestrians']:
                bbox_2d = self._project_3d_to_2d(p, cam_config)
                if bbox_2d:
                    x_center, y_center, width, height = bbox_2d
                    x_center /= w
                    y_center /= h
                    width /= w
                    height /= h
                    if 0 < x_center < 1 and 0 < y_center < 1:
                        f.write(f"{p['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def _project_3d_to_2d(self, obj: Dict, cam_config: CameraConfig) -> Optional[Tuple[float, float, float, float]]:
        """پروجکشن ساده 3D به 2D (برای داده سنتتیک)"""
        # این یک تقدیر ساده است - در عمل باید از ماتریس کالیبراسیون دوربین استفاده شود
        loc = obj['location']
        cam_loc = cam_config.location
        
        # Relative position
        dx = loc['x'] - cam_loc[0]
        dy = loc['y'] - cam_loc[1]
        dz = loc['z'] - cam_loc[2]
        
        # Simple pinhole camera model
        fov_rad = math.radians(cam_config.fov)
        focal_length = (cam_config.resolution[0] / 2) / math.tan(fov_rad / 2)
        
        # Camera rotation
        cam_yaw = math.radians(cam_config.rotation[1])
        
        # Rotate to camera frame
        rx = dx * math.cos(cam_yaw) + dy * math.sin(cam_yaw)
        ry = -dx * math.sin(cam_yaw) + dy * math.cos(cam_yaw)
        rz = dz
        
        if rz <= 0:  # Behind camera
            return None
        
        # Project
        u = focal_length * rx / rz + cam_config.resolution[0] / 2
        v = focal_length * ry / rz + cam_config.resolution[1] / 2
        
        # Estimate bbox size based on distance and class
        extent = obj.get('extent', {'x': 1.5, 'y': 1.5, 'z': 1.5})
        bbox_w = focal_length * (extent['x'] * 2) / rz
        bbox_h = focal_length * (extent['z'] * 2) / rz
        
        return (u, v, bbox_w, bbox_h)
    
    def _save_tracking_annotation(self, frame_data: Dict, split: str):
        """ذخیره برچسب ردیابی (MOT format)"""
        frame_id = frame_data['frame_id']
        track_path = self.output_dir / 'tracking' / f"{split}_tracks.txt"
        
        with open(track_path, 'a') as f:
            for v in frame_data['ground_truth']['vehicles']:
                bbox_2d = self._project_3d_to_2d(v, self.camera_configs[0])  # Use first camera
                if bbox_2d:
                    x, y, w, h = bbox_2d
                    # MOT format: frame, id, x, y, w, h, conf, class, visibility
                    f.write(f"{frame_id},{v['id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{v['class_id']},1\n")
    
    def _make_serializable(self, obj):
        """تبدیل numpy arrays به list برای JSON"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _clear_traffic(self):
        """پاک کردن ترافیک قبلی"""
        for v in self.vehicles:
            if v['actor'].is_alive:
                v['actor'].destroy()
        for w in self.walkers:
            if w.is_alive:
                w.destroy()
        for c in self.walker_controllers:
            if c.is_alive:
                c.destroy()
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []
    
    def run_generation(self, 
                       num_frames: int = 50000,
                       frames_per_hour: int = 2000,
                       val_split: float = 0.1,
                       host: str = 'localhost',
                       port: int = 2000):
        """اجرای کامل تولید داده"""
        print(f"🚀 Starting synthetic data generation: {num_frames} frames")
        
        # Connect
        self.connect_carla(host, port)
        
        # Setup cameras
        self.setup_cameras()
        
        # Generate for each hour
        hours = list(self.HOURLY_TRAFFIC_PATTERN.keys())
        frames_per_hour_actual = num_frames // len(hours)
        
        for hour in hours:
            self.current_hour = hour
            print(f"\n📅 Generating hour {hour}:00 ({frames_per_hour_actual} frames)")
            
            # Spawn traffic for this hour
            self.spawn_traffic(base_vehicle_count=150)
            
            # Let traffic stabilize
            for _ in range(50):
                self.world.tick()
            
            # Capture frames
            for frame_idx in range(frames_per_hour_actual):
                split = 'val' if random.random() < val_split else 'train'
                
                # Inject anomalies periodically
                if frame_idx % 100 == 0:
                    self.inject_anomalies(intensity=0.15)
                
                self.capture_frame(split)
                
                # Progress
                if frame_idx % 500 == 0:
                    print(f"  Frame {frame_idx}/{frames_per_hour_actual} | "
                          f"Total: {self.stats['frames_captured']} | "
                          f"Anomalies: {sum(self.stats['anomalies_injected'].values())}")
        
        # Save dataset.yaml for YOLO
        self._save_yolo_dataset_yaml()
        
        # Save statistics
        self._save_statistics()
        
        print(f"\n✅ Generation complete!")
        print(f"   Total frames: {self.stats['frames_captured']}")
        print(f"   Total anomalies: {sum(self.stats['anomalies_injected'].values())}")
        print(f"   Output: {self.output_dir}")
    
    def _save_yolo_dataset_yaml(self):
        """تولید dataset.yaml برای YOLO"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {v: k for k, v in self.IRAN_CLASSES.items()},
            'nc': len(self.IRAN_CLASSES),
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)
        print(f"✅ YOLO dataset.yaml saved to {yaml_path}")
    
    def _save_statistics(self):
        """ذخیره آمار تولید"""
        stats_path = self.output_dir / 'generation_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"✅ Statistics saved to {stats_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Isfahan synthetic traffic data')
    parser.add_argument('--frames', type=int, default=50000, help='Number of frames to generate')
    parser.add_argument('--host', default='localhost', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--output', default='data/synthetic', help='Output directory')
    args = parser.parse_args()
    
    generator = IsfahanSyntheticDataGenerator()
    generator.run_generation(
        num_frames=args.frames,
        host=args.host,
        port=args.port
    )


if __name__ == '__main__':
    main()