#!/usr/bin/env python3
"""
Isfahan DeepStream Pipeline - Hardware Accelerated Multi-Camera Inference
YOLOv10 Detection + BoT-SORT Tracking + ST-GCN Anomaly Detection
Optimized for Jetson Orin AGX 64GB with TensorRT INT8
"""

import sys
import os
import gi
import argparse
import logging
import json
import time
import threading
import queue
import signal
from pathlib import Path
from typing import Dict, List, Optional

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GObject

# Initialize GStreamer
Gst.init(None)
GObject.threads_init()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IsfahanDeepStreamPipeline:
    """
    Multi-camera DeepStream pipeline for Isfahan traffic intersection
    8 cameras → YOLOv10 Detection → BoT-SORT Tracking → ST-GCN Anomaly → MQTT Output
    """
    
    def __init__(self, config_path: str, cameras: int = 8, 
                 triton_url: str = "localhost:8001", 
                 mqtt_broker: str = "mqtt://localhost:1883",
                 log_level: str = "INFO"):
        self.config_path = config_path
        self.num_cameras = cameras
        self.triton_url = triton_url
        self.mqtt_broker = mqtt_broker
        self.log_level = log_level
        
        # Load intersection config
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.intersection = self.config['intersection']
        self.cameras_config = self.intersection['cameras']
        
        # Pipeline components
        self.pipeline = None
        self.loop = None
        self.running = False
        
        # Camera sources
        self.rtsp_sources = []
        for cam in self.cameras_config:
            self.rtsp_sources.append({
                'id': cam['id'],
                'name': cam['name_fa'],
                'uri': cam['rtsp'],
                'position': cam['position'],
                'zone': cam['position'],  # approach zone
            })
        
        # Model paths (TensorRT engines)
        self.models = {
            'detection': '/app/weights/detection/yolov10n_isfahan_int8.engine',
            'reid': '/app/weights/tracking/osnet_x1_0_isfahan.engine',
            'anomaly': '/app/weights/anomaly/stgcn_isfahan.onnx',
        }
        
        # Class mapping
        self.class_names = [
            'sedan_iran', 'pickup_iran', 'motorcycle_iran', 'bus_brt',
            'bus_city', 'truck_heavy', 'van_transport', 'bicycle',
            'pedestrian', 'traffic_light', 'traffic_sign', 'crosswalk'
        ]
        
        # Zone definitions for tracking
        self.zones = self._define_zones()
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_total': 0,
            'tracks_active': 0,
            'anomalies_detected': 0,
            'start_time': time.time(),
            'per_camera': {cam['id']: {'fps': 0, 'detections': 0, 'tracks': 0} 
                          for cam in self.rtsp_sources},
        }
        self.stats_lock = threading.Lock()
    
    def _define_zones(self) -> Dict:
        """تعریف نواحی تقاطع برای ردیابی و ناهنجاری"""
        return {
            'north_approach': {
                'polygon': [(100, -20), (140, -20), (140, 20), (100, 20)],
                'lanes': [1, 2, 3, 4],
                'direction': 'southbound',
            },
            'south_approach': {
                'polygon': [(-140, -20), (-100, -20), (-100, 20), (-140, 20)],
                'lanes': [1, 2, 3, 4],
                'direction': 'northbound',
            },
            'east_approach': {
                'polygon': [(-20, 100), (20, 100), (20, 140), (-20, 140)],
                'lanes': [1, 2, 3],
                'direction': 'westbound',
            },
            'west_approach': {
                'polygon': [(-20, -140), (20, -140), (20, -100), (-20, -100)],
                'lanes': [1, 2, 3],
                'direction': 'eastbound',
            },
            'intersection_center': {
                'polygon': [(-15, -15), (15, -15), (15, 15), (-15, 15)],
                'lanes': [],
                'direction': 'none',
            },
            'crosswalk_north': {
                'polygon': [(100, -15), (140, -15), (140, 15), (100, 15)],
                'type': 'crosswalk',
            },
            'crosswalk_south': {
                'polygon': [(-140, -15), (-100, -15), (-100, 15), (-140, 15)],
                'type': 'crosswalk',
            },
            'crosswalk_east': {
                'polygon': [(-15, 100), (15, 100), (15, 140), (-15, 140)],
                'type': 'crosswalk',
            },
            'crosswalk_west': {
                'polygon': [(-15, -140), (15, -140), (15, -100), (-15, -100)],
                'type': 'crosswalk',
            },
            'brt_lane_north': {
                'polygon': [(120, -20), (125, -20), (125, 20), (120, 20)],
                'type': 'brt_lane',
                'restricted_to': ['bus_brt'],
            },
            'brt_lane_south': {
                'polygon': [(-125, -20), (-120, -20), (-120, 20), (-125, 20)],
                'type': 'brt_lane',
                'restricted_to': ['bus_brt'],
            },
        }
    
    def build_pipeline(self) -> Gst.Pipeline:
        """ساخت پایپ‌لاین GStreamer کامل"""
        pipeline_str = self._generate_pipeline_string()
        logger.info("Building pipeline...")
        logger.debug(f"Pipeline: {pipeline_str}")
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        if not self.pipeline:
            raise RuntimeError("Failed to parse pipeline")
        
        # Setup bus message handling
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        # Setup probe for inference results
        self._setup_inference_probes()
        
        return self.pipeline
    
    def _generate_pipeline_string(self) -> str:
        """تولید رشته پایپ‌لاین GStreamer"""
        pipeline_parts = []
        
        # 1. RTSP Sources (8 cameras)
        for i, cam in enumerate(self.rtsp_sources):
            source = (
                f"rtspsrc location={cam['uri']} latency=100 "
                f"protocols=tcp ! rtph264depay ! h264parse ! nvv4l2decoder "
                f"enable-max-performance=1 ! "
                f"queue name=src_queue_{i} ! "
                f"nvstreammux.sink_{i}"
            )
            pipeline_parts.append(source)
        
        # 2. Stream Muxer (batch 8 streams)
        muxer = (
            "nvstreammux name=nvstreammux "
            "batch-size=8 "
            "width=1920 height=1080 "
            "batched-push-timeout=40000 "
            "live-source=1 "
            "attach-sys-ts=1 "
            "! "
        )
        pipeline_parts.append(muxer)
        
        # 3. Primary GIE (YOLOv10 Detection via Triton)
        pgie = (
            "nvinfer name=pgie_yolo10 "
            f"config-file-path=/app/pipeline/configs/pgie_yolo10_isfahan.txt "
            "unique-id=1 "
            "! "
        )
        pipeline_parts.append(pgie)
        
        # 4. Tracker (BoT-SORT via NvDCF + ReID)
        tracker = (
            "nvtracker name=tracker_botsort "
            "ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvdcftracker.so "
            "ll-config-file=/app/pipeline/configs/tracker_botsort_isfahan.yml "
            "enable-batch-process=1 "
            "display-tracking-id=1 "
            "! "
        )
        pipeline_parts.append(tracker)
        
        # 5. Secondary GIE - ReID for tracking association
        sgie_reid = (
            "nvinfer name=sgie_reid "
            f"config-file-path=/app/pipeline/configs/sgie_reid_isfahan.txt "
            "unique-id=2 "
            "process-mode=2 "  # operate on tracked objects
            "operate-on-gie-id=1 "
            "! "
        )
        pipeline_parts.append(sgie_reid)
        
        # 6. Secondary GIE - Anomaly Classification (ST-GCN)
        sgie_anomaly = (
            "nvinfer name=sgie_anomaly "
            f"config-file-path=/app/pipeline/configs/sgie_anomaly_isfahan.txt "
            "unique-id=3 "
            "process-mode=2 "
            "operate-on-gie-id=1 "
            "! "
        )
        pipeline_parts.append(sgie_anomaly)
        
        # 7. Tiler (for display/monitoring)
        tiler = (
            "nvmultistreamtiler name=tiler "
            "rows=2 columns=4 "
            "width=1920 height=1080 "
            "! "
        )
        pipeline_parts.append(tiler)
        
        # 8. OSD (On-Screen Display)
        osd = (
            "nvdsosd name=osd "
            "process-mode=0 "
            "display-text=1 "
            "display-bbox=1 "
            "display-mask=0 "
            "! "
        )
        pipeline_parts.append(osd)
        
        # 9. Sink (multiple: fakesink for headless, RTSP for monitoring)
        sink = (
            "tee name=t "
            "t. ! queue ! fakesink async=false "
            "t. ! queue ! nvv4l2h264enc bitrate=4000000 ! "
            "rtph264pay ! udpsink host=224.0.0.1 port=5000 auto-multicast=true "
            "t. ! queue ! nvv4l2h264enc bitrate=2000000 ! "
            "h264parse ! rtph264pay pt=96 ! "
            "udpsink host=127.0.0.1 port=5004 buffer-size=65536 "
        )
        pipeline_parts.append(sink)
        
        return " ".join(pipeline_parts)
    
    def _setup_inference_probes(self):
        """تنظیم probe برای دریافت نتایج استنتاج"""
        # Probe on PGIE (detection) output
        pgie = self.pipeline.get_by_name("pgie_yolo10")
        if pgie:
            pad = pgie.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self._pgie_probe_callback, "detection")
        
        # Probe on Tracker output
        tracker = self.pipeline.get_by_name("tracker_botsort")
        if tracker:
            pad = tracker.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self._tracker_probe_callback, "tracking")
        
        # Probe on Anomaly SGIE output
        sgie_anomaly = self.pipeline.get_by_name("sgie_anomaly")
        if sgie_anomaly:
            pad = sgie_anomaly.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, self._anomaly_probe_callback, "anomaly")
    
    def _pgie_probe_callback(self, pad, info, probe_type):
        """Callback برای نتایج تشخیص"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        # Parse NvDsBatchMeta
        batch_meta = self._get_batch_meta(gst_buffer)
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        frame_meta_list = batch_meta.frame_meta_list
        while frame_meta_list:
            frame_meta = frame_meta_list.data
            if frame_meta:
                self._process_detection_frame(frame_meta)
            frame_meta_list = frame_meta_list.next
        
        return Gst.PadProbeReturn.OK
    
    def _tracker_probe_callback(self, pad, info, probe_type):
        """Callback برای نتایج ردیابی"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = self._get_batch_meta(gst_buffer)
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        frame_meta_list = batch_meta.frame_meta_list
        while frame_meta_list:
            frame_meta = frame_meta_list.data
            if frame_meta:
                self._process_tracking_frame(frame_meta)
            frame_meta_list = frame_meta_list.next
        
        return Gst.PadProbeReturn.OK
    
    def _anomaly_probe_callback(self, pad, info, probe_type):
        """Callback برای نتایج ناهنجاری"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = self._get_batch_meta(gst_buffer)
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        frame_meta_list = batch_meta.frame_meta_list
        while frame_meta_list:
            frame_meta = frame_meta_list.data
            if frame_meta:
                self._process_anomaly_frame(frame_meta)
            frame_meta_list = frame_meta_list.next
        
        return Gst.PadProbeReturn.OK
    
    def _get_batch_meta(self, gst_buffer):
        """استخراج NvDsBatchMeta از بافر"""
        try:
            import pyds
            return pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        except Exception as e:
            logger.debug(f"Could not get batch meta: {e}")
            return None
    
    def _process_detection_frame(self, frame_meta):
        """پردازش نتایج تشخیص برای یک فریم"""
        import pyds
        
        frame_number = frame_meta.frame_num
        source_id = frame_meta.pad_index
        cam_info = self.rtsp_sources[source_id] if source_id < len(self.rtsp_sources) else None
        
        obj_meta_list = frame_meta.obj_meta_list
        detections = []
        
        while obj_meta_list:
            obj_meta = obj_meta_list.data
            if obj_meta:
                class_id = obj_meta.class_id
                confidence = obj_meta.confidence
                
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    
                    # Bounding box
                    rect = obj_meta.rect_params
                    bbox = {
                        'x': rect.left,
                        'y': rect.top,
                        'width': rect.width,
                        'height': rect.height,
                    }
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'track_id': obj_meta.object_id if obj_meta.object_id != 0xFFFFFFFF else None,
                    })
                    
                    # Update stats
                    with self.stats_lock:
                        self.stats['detections_total'] += 1
                        if cam_info:
                            self.stats['per_camera'][cam_info['id']]['detections'] += 1
            
            obj_meta_list = obj_meta_list.next
        
        # Publish to MQTT
        if detections and self.mqtt_connected:
            self._publish_detections(cam_info, frame_number, detections)
    
    def _process_tracking_frame(self, frame_meta):
        """پردازش نتایج ردیابی"""
        import pyds
        
        frame_number = frame_meta.frame_num
        source_id = frame_meta.pad_index
        cam_info = self.rtsp_sources[source_id] if source_id < len(self.rtsp_sources) else None
        
        obj_meta_list = frame_meta.obj_meta_list
        tracks = []
        
        while obj_meta_list:
            obj_meta = obj_meta_list.data
            if obj_meta and obj_meta.object_id != 0xFFFFFFFF:
                track_id = obj_meta.object_id
                class_id = obj_meta.class_id
                
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    
                    rect = obj_meta.rect_params
                    bbox = {
                        'x': rect.left,
                        'y': rect.top,
                        'width': rect.width,
                        'height': rect.height,
                    }
                    
                    # Get zone
                    center_x = rect.left + rect.width / 2
                    center_y = rect.top + rect.height / 2
                    zone = self._get_zone(center_x, center_y)
                    
                    # Get ReID features (from user_meta)
                    reid_features = self._get_reid_features(obj_meta)
                    
                    tracks.append({
                        'track_id': track_id,
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': bbox,
                        'zone': zone,
                        'reid_features': reid_features,
                        'age': obj_meta.tracker_confidence if hasattr(obj_meta, 'tracker_confidence') else 0,
                    })
                    
                    with self.stats_lock:
                        self.stats['tracks_active'] += 1
                        if cam_info:
                            self.stats['per_camera'][cam_info['id']]['tracks'] += 1
            
            obj_meta_list = obj_meta_list.next
        
        if tracks and self.mqtt_connected:
            self._publish_tracks(cam_info, frame_number, tracks)
    
    def _process_anomaly_frame(self, frame_meta):
        """پردازش نتایج تشخیص ناهنجاری"""
        import pyds
        
        frame_number = frame_meta.frame_num
        source_id = frame_meta.pad_index
        cam_info = self.rtsp_sources[source_id] if source_id < len(self.rtsp_sources) else None
        
        obj_meta_list = frame_meta.obj_meta_list
        
        while obj_meta_list:
            obj_meta = obj_meta_list.data
            if obj_meta and obj_meta.object_id != 0xFFFFFFFF:
                # Check for anomaly classification in user_meta
                anomaly_info = self._get_anomaly_classification(obj_meta)
                if anomaly_info and anomaly_info['anomaly_id'] != 0:  # 0 = normal
                    track_id = obj_meta.object_id
                    class_id = obj_meta.class_id
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                    
                    anomaly_data = {
                        'track_id': track_id,
                        'class_name': class_name,
                        'anomaly_type': anomaly_info['anomaly_name'],
                        'anomaly_id': anomaly_info['anomaly_id'],
                        'confidence': anomaly_info['confidence'],
                        'severity': anomaly_info['severity'],
                        'camera_id': cam_info['id'] if cam_info else f'cam_{source_id}',
                        'camera_name': cam_info['name'] if cam_info else f'Camera {source_id}',
                        'zone': anomaly_info.get('zone', 'unknown'),
                        'timestamp': time.time(),
                        'frame_number': frame_number,
                    }
                    
                    with self.stats_lock:
                        self.stats['anomalies_detected'] += 1
                    
                    # Publish anomaly alert
                    if self.mqtt_connected:
                        self._publish_anomaly(anomaly_data)
            
            obj_meta_list = obj_meta_list.next
    
    def _get_zone(self, x: float, y: float) -> str:
        """تعیین ناحیه بر اساس مختصات"""
        for zone_name, zone_info in self.zones.items():
            polygon = zone_info['polygon']
            if self._point_in_polygon(x, y, polygon):
                return zone_name
        return 'unknown'
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple]) -> bool:
        """بررسی نقطه در مضلع (Ray casting)"""
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside
    
    def _get_reid_features(self, obj_meta) -> Optional[List[float]]:
        """استخراج ویژگی ReID از user_meta"""
        import pyds
        
        user_meta_list = obj_meta.obj_user_meta_list
        while user_meta_list:
            user_meta = user_meta_list.data
            if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                if tensor_meta and tensor_meta.num_output_layers > 0:
                    # Get first output layer (embedding)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    if layer and layer.buffer:
                        # Convert to numpy array
                        import numpy as np
                        ptr = pyds.get_ptr(layer.buffer)
                        embedding = np.ctypeslib.as_array(
                            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                            shape=(layer.dims.numElements,)
                        )
                        return embedding.tolist()
            user_meta_list = user_meta_list.next
        return None
    
    def _get_anomaly_classification(self, obj_meta) -> Optional[Dict]:
        """استخراج کلاس ناهنجاری از user_meta"""
        import pyds
        
        user_meta_list = obj_meta.obj_user_meta_list
        while user_meta_list:
            user_meta = user_meta_list.data
            if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                if tensor_meta and tensor_meta.unique_id == 3:  # Anomaly SGIE
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    if layer and layer.buffer:
                        import numpy as np
                        ptr = pyds.get_ptr(layer.buffer)
                        logits = np.ctypeslib.as_array(
                            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)),
                            shape=(layer.dims.numElements,)
                        )
                        # Softmax
                        probs = np.exp(logits) / np.sum(np.exp(logits))
                        anomaly_id = int(np.argmax(probs))
                        confidence = float(probs[anomaly_id])
                        
                        # Anomaly names from config
                        anomaly_names = [
                            'normal', 'motorcycle_wrong_lane', 'red_light_violation',
                            'illegal_uturn', 'stop_on_crosswalk', 'wrong_way_driving',
                            'pedestrian_jaywalking', 'bus_lane_violation',
                            'truck_in_car_lane', 'sudden_lane_change', 'speeding'
                        ]
                        
                        severity_map = {
                            0: 0, 1: 3, 2: 10, 3: 5, 4: 4, 5: 10,
                            6: 3, 7: 4, 8: 6, 9: 5, 10: 7
                        }
                        
                        return {
                            'anomaly_id': anomaly_id,
                            'anomaly_name': anomaly_names[anomaly_id] if anomaly_id < len(anomaly_names) else 'unknown',
                            'confidence': confidence,
                            'severity': severity_map.get(anomaly_id, 0),
                        }
            user_meta_list = user_meta_list.next
        return None
    
    def _publish_detections(self, cam_info, frame_number, detections):
        """انتشار تشخیص‌ها به MQTT"""
        topic = f"{self.intersection['integration']['mqtt_topic_prefix']}/detections"
        payload = {
            'intersection_id': self.intersection['id'],
            'camera_id': cam_info['id'],
            'camera_name': cam_info['name'],
            'camera_zone': cam_info['zone'],
            'frame_number': frame_number,
            'timestamp': time.time(),
            'detections': detections,
        }
        self._mqtt_publish(topic, payload)
    
    def _publish_tracks(self, cam_info, frame_number, tracks):
        """انتشار ردیابی‌ها به MQTT"""
        topic = f"{self.intersection['integration']['mqtt_topic_prefix']}/tracks"
        payload = {
            'intersection_id': self.intersection['id'],
            'camera_id': cam_info['id'],
            'camera_name': cam_info['name'],
            'camera_zone': cam_info['zone'],
            'frame_number': frame_number,
            'timestamp': time.time(),
            'tracks': tracks,
        }
        self._mqtt_publish(topic, payload)
    
    def _publish_anomaly(self, anomaly_data):
        """انتشار هشدار ناهنجاری به MQTT"""
        topic = f"{self.intersection['integration']['mqtt_topic_prefix']}/anomalies"
        payload = {
            'intersection_id': self.intersection['id'],
            'alert_id': f"anom_{anomaly_data['track_id']}_{int(anomaly_data['timestamp'])}",
            'timestamp': anomaly_data['timestamp'],
            'anomaly': anomaly_data,
        }
        self._mqtt_publish(topic, payload, qos=1, retain=False)
    
    def _mqtt_publish(self, topic: str, payload: Dict, qos: int = 0, retain: bool = False):
        """انتشار به MQTT"""
        if self.mqtt_client and self.mqtt_connected:
            try:
                self.mqtt_client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=qos, retain=retain)
            except Exception as e:
                logger.error(f"MQTT publish error: {e}")
    
    def setup_mqtt(self):
        """راه‌اندازی کلاینت MQTT"""
        try:
            import paho.mqtt.client as mqtt
            
            self.mqtt_client = mqtt.Client(client_id=f"isfahan-deepstream-{os.getpid()}")
            
            # TLS
            if self.mqtt_broker.startswith("mqtts://") or self.mqtt_broker.startswith("ssl://"):
                self.mqtt_client.tls_set(
                    ca_certs="/certs/ca.crt",
                    certfile="/certs/client.crt",
                    keyfile="/certs/client.key",
                )
                self.mqtt_client.tls_insecure_set(False)
            
            # Callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.on_publish = self._on_mqtt_publish
            
            # Parse broker URL
            import urllib.parse
            parsed = urllib.parse.urlparse(self.mqtt_broker)
            host = parsed.hostname or "localhost"
            port = parsed.port or (8883 if parsed.scheme == "mqtts" else 1883)
            
            # Credentials
            if parsed.username:
                self.mqtt_client.username_pw_set(parsed.username, parsed.password)
            
            # Connect
            self.mqtt_client.connect_async(host, port, keepalive=60)
            self.mqtt_client.loop_start()
            
            logger.info(f"MQTT connecting to {host}:{port}...")
            
        except Exception as e:
            logger.error(f"MQTT setup failed: {e}")
            self.mqtt_client = None
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mqtt_connected = True
            logger.info("✅ MQTT connected")
            
            # Subscribe to signal control commands
            topic = f"{self.intersection['integration']['mqtt_topic_prefix']}/signal_commands"
            client.subscribe(topic, qos=1)
        else:
            logger.error(f"MQTT connection failed: {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        logger.warning(f"MQTT disconnected: {rc}")
    
    def _on_mqtt_publish(self, client, userdata, mid):
        pass  # logger.debug(f"MQTT published: {mid}")
    
    def start(self):
        """شروع پایپ‌لاین"""
        if not self.pipeline:
            self.build_pipeline()
        
        # Setup MQTT
        self.setup_mqtt()
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start pipeline")
        
        self.running = True
        self.loop = GLib.MainLoop()
        
        # Start stats reporter
        stats_thread = threading.Thread(target=self._report_stats, daemon=True)
        stats_thread.start()
        
        # Handle signals
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Pipeline started. Press Ctrl+C to stop.")
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """توقف پایپ‌لاین"""
        logger.info("Stopping pipeline...")
        self.running = False
        
        if self.loop:
            self.loop.quit()
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        logger.info("✅ Pipeline stopped")
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.stop()
    
    def _report_stats(self):
        """گزارش آمار دوره‌ای"""
        while self.running:
            time.sleep(30)
            with self.stats_lock:
                elapsed = time.time() - self.stats['start_time']
                fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
                
                logger.info(
                    f"📊 Stats: Frames={self.stats['frames_processed']}, "
                    f"FPS={fps:.1f}, Detections={self.stats['detections_total']}, "
                    f"Tracks={self.stats['tracks_active']}, Anomalies={self.stats['anomalies_detected']}"
                )
                
                for cam_id, cam_stats in self.stats['per_camera'].items():
                    logger.info(f"  {cam_id}: fps={cam_stats['fps']:.1f}, "
                              f"det={cam_stats['detections']}, trk={cam_stats['tracks']}")


def main():
    parser = argparse.ArgumentParser(description='Isfahan DeepStream Pipeline')
    parser.add_argument('--config', default='/app/configs/intersection.yaml', help='Intersection config')
    parser.add_argument('--cameras', type=int, default=8, help='Number of cameras')
    parser.add_argument('--triton-url', default='localhost:8001', help='Triton server URL')
    parser.add_argument('--mqtt-broker', default='mqtt://localhost:1883', help='MQTT broker URL')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create and run pipeline
    pipeline = IsfahanDeepStreamPipeline(
        config_path=args.config,
        cameras=args.cameras,
        triton_url=args.triton_url,
        mqtt_broker=args.mqtt_broker,
        log_level=args.log_level,
    )
    
    try:
        pipeline.start()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()