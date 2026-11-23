#!/usr/bin/env python3
"""
MQTT Bridge for Isfahan Smart Intersection
Bridges DeepStream pipeline ↔ Isfahan UTC/SCADA/CMS systems
Supports TLS, QoS, message routing, and protocol translation
"""

import os
import json
import logging
import asyncio
import signal
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import paho.mqtt.client as mqtt
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MQTTConfig:
    broker: str
    port: int
    username: Optional[str]
    password: Optional[str]
    topic_prefix: str
    client_id: str
    tls_ca_cert: Optional[str]
    tls_client_cert: Optional[str]
    tls_client_key: Optional[str]
    tls_insecure: bool
    heartbeat_interval: int
    reconnect_delay: int
    max_queued_messages: int


@dataclass
class BridgeRoute:
    source_topic: str
    target_topic: str
    transform: Optional[str] = None  # Python expression for transformation
    qos: int = 1
    filter_expr: Optional[str] = None  # JMESPath filter


class IsfahanMQTTBridge:
    """
    MQTT Bridge connecting DeepStream pipeline to Isfahan municipal systems:
    - UTC (Urban Traffic Control)
    - SCADA (Supervisory Control and Data Acquisition)
    - CMS (Central Management System)
    """
    
    def __init__(self, config_path: str = "/app/configs/intersection.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.mqtt_config = self._parse_mqtt_config()
        
        # MQTT Clients
        self.local_client: Optional[mqtt.Client] = None  # DeepStream ↔ Bridge
        self.remote_client: Optional[mqtt.Client] = None  # Bridge ↔ UTC/SCADA
        
        # Routing rules
        self.routes: List[BridgeRoute] = self._define_routes()
        
        # Message queues for async processing
        self.local_to_remote_queue: asyncio.Queue = asyncio.Queue(maxsize=self.mqtt_config.max_queued_messages)
        self.remote_to_local_queue: asyncio.Queue = asyncio.Queue(maxsize=self.mqtt_config.max_queued_messages)
        
        # State
        self.running = False
        self.local_connected = False
        self.remote_connected = False
        self.last_heartbeat = time.time()
        
        # Statistics
        self.stats = {
            'messages_received_local': 0,
            'messages_received_remote': 0,
            'messages_forwarded': 0,
            'messages_dropped': 0,
            'errors': 0,
            'start_time': time.time(),
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _parse_mqtt_config(self) -> MQTTConfig:
        integration = self.config['intersection']['integration']
        
        # Parse broker URL
        broker_url = integration.get('mqtt_broker', 'mqtt://localhost:1883')
        if broker_url.startswith('mqtts://') or broker_url.startswith('ssl://'):
            default_port = 8883
            use_tls = True
        else:
            default_port = 1883
            use_tls = False
        
        import urllib.parse
        parsed = urllib.parse.urlparse(broker_url)
        host = parsed.hostname or 'localhost'
        port = parsed.port or default_port
        
        return MQTTConfig(
            broker=host,
            port=port,
            username=parsed.username or os.getenv('MQTT_USERNAME'),
            password=parsed.password or os.getenv('MQTT_PASSWORD'),
            topic_prefix=integration.get('mqtt_topic_prefix', 'isfahan/traffic/chaharbag_azadi'),
            client_id=os.getenv('MQTT_CLIENT_ID', f'isfahan-bridge-{os.getpid()}'),
            tls_ca_cert=os.getenv('TLS_CA_CERT', '/certs/ca.crt') if use_tls else None,
            tls_client_cert=os.getenv('TLS_CLIENT_CERT', '/certs/client.crt') if use_tls else None,
            tls_client_key=os.getenv('TLS_CLIENT_KEY', '/certs/client.key') if use_tls else None,
            tls_insecure=os.getenv('TLS_INSECURE', 'false').lower() == 'true',
            heartbeat_interval=int(os.getenv('HEARTBEAT_INTERVAL', '30')),
            reconnect_delay=int(os.getenv('RECONNECT_DELAY', '5')),
            max_queued_messages=int(os.getenv('MAX_QUEUED_MESSAGES', '10000')),
        )
    
    def _define_routes(self) -> List[BridgeRoute]:
        """تعریف مسیرهای مسیریابی پیام‌ها"""
        prefix = self.mqtt_config.topic_prefix
        
        return [
            # DeepStream → UTC/SCADA/CMS
            BridgeRoute(
                source_topic=f"{prefix}/detections",
                target_topic=f"utc/traffic/detections",
                qos=0,  # High frequency, best effort
            ),
            BridgeRoute(
                source_topic=f"{prefix}/tracks",
                target_topic=f"utc/traffic/tracks",
                qos=0,
            ),
            BridgeRoute(
                source_topic=f"{prefix}/anomalies",
                target_topic=f"scada/alerts/traffic_anomalies",
                qos=1,  # Critical alerts, guaranteed delivery
                transform="add_severity_priority",
            ),
            BridgeRoute(
                source_topic=f"{prefix}/anomalies",
                target_topic=f"cms/events/traffic_violations",
                qos=1,
                transform="format_cms_violation",
            ),
            BridgeRoute(
                source_topic=f"{prefix}/signal_status",
                target_topic=f"utc/signals/status",
                qos=1,
            ),
            BridgeRoute(
                source_topic=f"{prefix}/health",
                target_topic=f"scada/monitoring/edge_health",
                qos=1,
            ),
            
            # UTC/SCADA → DeepStream
            BridgeRoute(
                source_topic=f"utc/signals/commands",
                target_topic=f"{prefix}/signal_commands",
                qos=1,
                transform="parse_signal_command",
            ),
            BridgeRoute(
                source_topic=f"scada/control/override",
                target_topic=f"{prefix}/control_override",
                qos=1,
            ),
            BridgeRoute(
                source_topic=f"cms/config/update",
                target_topic=f"{prefix}/config_update",
                qos=1,
            ),
        ]
    
    def _setup_local_client(self):
        """راه‌اندازی کلاینت MQTT محلی (DeepStream ↔ Bridge)"""
        self.local_client = mqtt.Client(client_id=f"{self.mqtt_config.client_id}-local")
        self.local_client.on_connect = self._on_local_connect
        self.local_client.on_disconnect = self._on_local_disconnect
        self.local_client.on_message = self._on_local_message
        self.local_client.on_publish = self._on_local_publish
        
        # Local broker (usually same machine or local network)
        local_broker = os.getenv('LOCAL_MQTT_BROKER', 'localhost')
        local_port = int(os.getenv('LOCAL_MQTT_PORT', '1883'))
        
        logger.info(f"Connecting to local MQTT broker: {local_broker}:{local_port}")
        self.local_client.connect_async(local_broker, local_port, keepalive=60)
        self.local_client.loop_start()
    
    def _setup_remote_client(self):
        """راه‌اندازی کلاینت MQTT راه‌دور (Bridge ↔ UTC/SCADA)"""
        self.remote_client = mqtt.Client(client_id=f"{self.mqtt_config.client_id}-remote")
        self.remote_client.on_connect = self._on_remote_connect
        self.remote_client.on_disconnect = self._on_remote_disconnect
        self.remote_client.on_message = self._on_remote_message
        self.remote_client.on_publish = self._on_remote_publish
        
        # TLS Configuration
        if self.mqtt_config.tls_ca_cert and Path(self.mqtt_config.tls_ca_cert).exists():
            self.remote_client.tls_set(
                ca_certs=self.mqtt_config.tls_ca_cert,
                certfile=self.mqtt_config.tls_client_cert,
                keyfile=self.mqtt_config.tls_client_key,
            )
            self.remote_client.tls_insecure_set(self.mqtt_config.tls_insecure)
            logger.info("TLS enabled for remote MQTT connection")
        
        # Authentication
        if self.mqtt_config.username:
            self.remote_client.username_pw_set(
                self.mqtt_config.username,
                self.mqtt_config.password
            )
        
        logger.info(f"Connecting to remote MQTT broker: {self.mqtt_config.broker}:{self.mqtt_config.port}")
        self.remote_client.connect_async(
            self.mqtt_config.broker, 
            self.mqtt_config.port, 
            keepalive=60
        )
        self.remote_client.loop_start()
    
    def _on_local_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.local_connected = True
            logger.info("✅ Local MQTT connected")
            
            # Subscribe to DeepStream topics
            for route in self.routes:
                if route.source_topic.startswith(self.mqtt_config.topic_prefix):
                    client.subscribe(route.source_topic, qos=route.qos)
                    logger.info(f"Subscribed to local topic: {route.source_topic}")
        else:
            logger.error(f"Local MQTT connection failed: {rc}")
    
    def _on_local_disconnect(self, client, userdata, rc):
        self.local_connected = False
        logger.warning(f"Local MQTT disconnected: {rc}")
    
    def _on_remote_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.remote_connected = True
            logger.info("✅ Remote MQTT (UTC/SCADA) connected")
            
            # Subscribe to UTC/SCADA topics
            for route in self.routes:
                if not route.source_topic.startswith(self.mqtt_config.topic_prefix):
                    client.subscribe(route.source_topic, qos=route.qos)
                    logger.info(f"Subscribed to remote topic: {route.source_topic}")
        else:
            logger.error(f"Remote MQTT connection failed: {rc}")
    
    def _on_remote_disconnect(self, client, userdata, rc):
        self.remote_connected = False
        logger.warning(f"Remote MQTT disconnected: {rc}")
    
    def _on_local_message(self, client, userdata, msg):
        """دریافت پیام از DeepStream → ارسال به UTC/SCADA"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode('utf-8'))
            
            self.stats['messages_received_local'] += 1
            
            # Find matching routes
            for route in self.routes:
                if self._topic_matches(topic, route.source_topic):
                    # Apply transformation
                    transformed = self._transform_message(payload, route.transform)
                    
                    # Add metadata
                    transformed['_bridge_meta'] = {
                        'source_topic': topic,
                        'target_topic': route.target_topic,
                        'bridge_timestamp': datetime.utcnow().isoformat() + 'Z',
                        'route_transform': route.transform,
                    }
                    
                    # Queue for async forwarding
                    try:
                        self.local_to_remote_queue.put_nowait((route, transformed))
                    except asyncio.QueueFull:
                        self.stats['messages_dropped'] += 1
                        logger.warning(f"Queue full, dropping message for {route.target_topic}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in local message: {e}")
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Error processing local message: {e}")
            self.stats['errors'] += 1
    
    def _on_remote_message(self, client, userdata, msg):
        """دریافت پیام از UTC/SCADA → ارسال به DeepStream"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode('utf-8'))
            
            self.stats['messages_received_remote'] += 1
            
            # Find matching routes
            for route in self.routes:
                if self._topic_matches(topic, route.source_topic):
                    transformed = self._transform_message(payload, route.transform)
                    
                    transformed['_bridge_meta'] = {
                        'source_topic': topic,
                        'target_topic': route.target_topic,
                        'bridge_timestamp': datetime.utcnow().isoformat() + 'Z',
                    }
                    
                    try:
                        self.remote_to_local_queue.put_nowait((route, transformed))
                    except asyncio.QueueFull:
                        self.stats['messages_dropped'] += 1
                        logger.warning(f"Queue full, dropping message for {route.target_topic}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in remote message: {e}")
            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Error processing remote message: {e}")
            self.stats['errors'] += 1
    
    def _on_local_publish(self, client, userdata, mid):
        pass
    
    def _on_remote_publish(self, client, userdata, mid):
        pass
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """بررسی تطابق تاپیک با الگو (پشتیبانی از wildcard)"""
        import fnmatch
        return fnmatch.fnmatch(topic, pattern)
    
    def _transform_message(self, payload: dict, transform: Optional[str]) -> dict:
        """اعمال تبدیل روی پیام"""
        if not transform:
            return payload
        
        try:
            if transform == "add_severity_priority":
                return self._add_severity_priority(payload)
            elif transform == "format_cms_violation":
                return self._format_cms_violation(payload)
            elif transform == "parse_signal_command":
                return self._parse_signal_command(payload)
            else:
                # Try to eval as Python expression
                # WARNING: Only use with trusted transforms
                local_vars = {'payload': payload, 'json': json, 'datetime': datetime}
                return eval(transform, {"__builtins__": {}}, local_vars)
        except Exception as e:
            logger.error(f"Transform '{transform}' failed: {e}")
            return payload
    
    def _add_severity_priority(self, payload: dict) -> dict:
        """اضافه کردن اولویت بر اساس شدت ناهنجاری"""
        severity_map = {
            'red_light_violation': 'critical',
            'wrong_way_driving': 'critical',
            'truck_in_car_lane': 'high',
            'speeding': 'high',
            'sudden_lane_change': 'medium',
            'bus_lane_violation': 'medium',
            'illegal_uturn': 'medium',
            'stop_on_crosswalk': 'low',
            'motorcycle_wrong_lane': 'low',
            'pedestrian_jaywalking': 'low',
        }
        
        anomaly_type = payload.get('anomaly', {}).get('anomaly_type', 'unknown')
        payload['priority'] = severity_map.get(anomaly_type, 'info')
        payload['severity_score'] = {
            'critical': 10, 'high': 7, 'medium': 5, 'low': 3, 'info': 1
        }.get(payload['priority'], 1)
        
        return payload
    
    def _format_cms_violation(self, payload: dict) -> dict:
        """فرمت‌بندی برای سیستم CMS شهرداری"""
        anomaly = payload.get('anomaly', {})
        
        return {
            'violation_id': payload.get('alert_id', f"viol_{int(time.time())}"),
            'violation_type': anomaly.get('anomaly_type', 'unknown'),
            'location': {
                'intersection': payload.get('intersection_id', 'chaharbag_azadi'),
                'camera': anomaly.get('camera_id', 'unknown'),
                'zone': anomaly.get('zone', 'unknown'),
            },
            'timestamp': anomaly.get('timestamp', time.time()),
            'vehicle_info': {
                'track_id': anomaly.get('track_id'),
                'class': anomaly.get('class_name', 'unknown'),
            },
            'evidence': {
                'confidence': anomaly.get('confidence', 0),
                'severity': anomaly.get('severity', 0),
            },
            'status': 'pending_review',
        }
    
    def _parse_signal_command(self, payload: dict) -> dict:
        """تفسیر دستور کنترل سیگنال از UTC"""
        # UTC sends: {intersection_id, plan_id, phase_durations, timestamp}
        return {
            'command': 'set_signal_plan',
            'intersection_id': payload.get('intersection_id'),
            'plan_id': payload.get('plan_id'),
            'phase_durations': payload.get('phase_durations', {}),
            'priority': payload.get('priority', 'normal'),
            'source': 'utc',
            'timestamp': payload.get('timestamp', time.time()),
        }
    
    async def _forward_local_to_remote(self):
        """Forward messages from local to remote broker"""
        while self.running:
            try:
                route, payload = await asyncio.wait_for(
                    self.local_to_remote_queue.get(), timeout=1.0
                )
                
                if self.remote_connected and self.remote_client:
                    try:
                        self.remote_client.publish(
                            route.target_topic,
                            json.dumps(payload, ensure_ascii=False),
                            qos=route.qos
                        )
                        self.stats['messages_forwarded'] += 1
                    except Exception as e:
                        logger.error(f"Failed to publish to remote: {e}")
                        self.stats['errors'] += 1
                        # Re-queue for retry
                        await self.local_to_remote_queue.put((route, payload))
                else:
                    # Remote not connected, re-queue
                    await self.local_to_remote_queue.put((route, payload))
                    await asyncio.sleep(1)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Forward local→remote error: {e}")
                self.stats['errors'] += 1
    
    async def _forward_remote_to_local(self):
        """Forward messages from remote to local broker"""
        while self.running:
            try:
                route, payload = await asyncio.wait_for(
                    self.remote_to_local_queue.get(), timeout=1.0
                )
                
                if self.local_connected and self.local_client:
                    try:
                        self.local_client.publish(
                            route.target_topic,
                            json.dumps(payload, ensure_ascii=False),
                            qos=route.qos
                        )
                        self.stats['messages_forwarded'] += 1
                    except Exception as e:
                        logger.error(f"Failed to publish to local: {e}")
                        self.stats['errors'] += 1
                        await self.remote_to_local_queue.put((route, payload))
                else:
                    await self.remote_to_local_queue.put((route, payload))
                    await asyncio.sleep(1)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Forward remote→local error: {e}")
                self.stats['errors'] += 1
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            await asyncio.sleep(self.mqtt_config.heartbeat_interval)
            
            if self.local_connected and self.local_client:
                heartbeat = {
                    'bridge_id': self.mqtt_config.client_id,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'status': 'healthy' if self.remote_connected else 'degraded',
                    'local_connected': self.local_connected,
                    'remote_connected': self.remote_connected,
                    'stats': self.stats,
                }
                self.local_client.publish(
                    f"{self.mqtt_config.topic_prefix}/bridge/heartbeat",
                    json.dumps(heartbeat),
                    qos=0
                )
            
            if self.remote_connected and self.remote_client:
                heartbeat = {
                    'bridge_id': self.mqtt_config.client_id,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'status': 'healthy' if self.local_connected else 'degraded',
                    'intersection': 'chaharbag_azadi',
                    'stats': self.stats,
                }
                self.remote_client.publish(
                    f"scada/monitoring/bridge_heartbeat",
                    json.dumps(heartbeat),
                    qos=0
                )
            
            self.last_heartbeat = time.time()
    
    async def _stats_loop(self):
        """Log statistics periodically"""
        while self.running:
            await asyncio.sleep(60)
            
            elapsed = time.time() - self.stats['start_time']
            rate = self.stats['messages_forwarded'] / elapsed if elapsed > 0 else 0
            
            logger.info(
                f"📊 Bridge Stats: "
                f"Local→Remote={self.stats['messages_received_local']}, "
                f"Remote→Local={self.stats['messages_received_remote']}, "
                f"Forwarded={self.stats['messages_forwarded']}, "
                f"Dropped={self.stats['messages_dropped']}, "
                f"Errors={self.stats['errors']}, "
                f"Rate={rate:.1f} msg/s, "
                f"Local={'🟢' if self.local_connected else '🔴'}, "
                f"Remote={'🟢' if self.remote_connected else '🔴'}"
            )
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def run(self):
        """اجرای اصلی Brücke"""
        self.running = True
        
        # Setup clients
        self._setup_local_client()
        self._setup_remote_client()
        
        # Wait for connections
        await asyncio.sleep(3)
        
        # Start forwarding tasks
        tasks = [
            asyncio.create_task(self._forward_local_to_remote()),
            asyncio.create_task(self._forward_remote_to_local()),
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._stats_loop()),
        ]
        
        logger.info("🚀 MQTT Bridge started")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """خاتمه بهینه"""
        logger.info("Shutting down MQTT Bridge...")
        self.running = False
        
        # Publish shutdown notice
        shutdown_msg = {
            'bridge_id': self.mqtt_config.client_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'status': 'shutdown',
        }
        
        if self.local_client:
            self.local_client.publish(
                f"{self.mqtt_config.topic_prefix}/bridge/status",
                json.dumps(shutdown_msg),
                qos=1, retain=True
            )
        
        if self.remote_client:
            self.remote_client.publish(
                f"scada/monitoring/bridge_status",
                json.dumps(shutdown_msg),
                qos=1, retain=True
            )
        
        # Stop loops
        if self.local_client:
            self.local_client.loop_stop()
            self.local_client.disconnect()
        
        if self.remote_client:
            self.remote_client.loop_stop()
            self.remote_client.disconnect()
        
        logger.info("✅ MQTT Bridge shutdown complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Isfahan MQTT Bridge')
    parser.add_argument('--config', default='/app/configs/intersection.yaml', help='Config file')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    bridge = IsfahanMQTTBridge(config_path=args.config)
    
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Bridge error: {e}")
        raise


if __name__ == '__main__':
    main()