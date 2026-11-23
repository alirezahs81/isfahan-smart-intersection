#!/usr/bin/env python3
"""
Health Monitor & Auto-Recovery for Isfahan Edge Deployment
Monitors GPU, cameras, containers, MQTT, and triggers auto-recovery
"""

import os
import json
import logging
import asyncio
import time
import signal
import subprocess
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import psutil
import pynvml
import docker
import requests
from aiohttp import web
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
HEALTH_CHECKS = Counter('isfahan_health_checks_total', 'Total health checks', ['component', 'status'])
GPU_MEMORY_USAGE = Gauge('isfahan_gpu_memory_usage_bytes', 'GPU memory usage', ['gpu'])
GPU_TEMPERATURE = Gauge('isfahan_gpu_temperature_celsius', 'GPU temperature', ['gpu'])
GPU_POWER = Gauge('isfahan_gpu_power_watts', 'GPU power draw', ['gpu'])
CAMERA_FPS = Gauge('isfahan_camera_fps', 'Camera FPS', ['camera_id'])
CAMERA_LATENCY = Gauge('isfahan_camera_latency_ms', 'Camera latency', ['camera_id'])
CONTAINER_STATUS = Gauge('isfahan_container_status', 'Container status (1=running, 0=stopped)', ['container'])
MQTT_CONNECTED = Gauge('isfahan_mqtt_connected', 'MQTT connection status', ['broker'])
ALERTS_SENT = Counter('isfahan_alerts_sent_total', 'Total alerts sent', ['severity', 'type'])
AUTO_RESTARTS = Counter('isfahan_auto_restarts_total', 'Total auto restarts', ['container'])


@dataclass
class HealthConfig:
    check_interval: int = 30
    gpu_memory_threshold: int = 90
    gpu_temp_threshold: int = 85
    gpu_power_threshold: int = 250
    camera_timeout: int = 10
    camera_fps_min: int = 20
    mqtt_heartbeat_timeout: int = 60
    triton_health_endpoint: str = "http://localhost:8000/v2/health/ready"
    deepstream_health_endpoint: str = "http://localhost:8080/health"
    alert_webhook: Optional[str] = None
    alert_webhook_timeout: int = 10
    auto_restart_services: List[str] = None
    max_restart_attempts: int = 3
    restart_cooldown: int = 300
    metrics_port: int = 9090


@dataclass
class ComponentHealth:
    name: str
    status: str  # healthy, degraded, critical, unknown
    last_check: float
    details: Dict[str, Any]
    consecutive_failures: int = 0


class IsfahanHealthMonitor:
    """
    Comprehensive health monitoring for Isfahan Smart Intersection edge deployment
    """
    
    def __init__(self, config_path: str = "/app/configs/intersection.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.health_config = self._parse_health_config()
        
        # Docker client
        self.docker_client = docker.from_env()
        
        # NVML initialization
        try:
            pynvml.nvmlInit()
            self.nvml_available = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized: {self.gpu_count} GPU(s) found")
        except Exception as e:
            logger.warning(f"NVML not available: {e}")
            self.nvml_available = False
            self.gpu_count = 0
        
        # Health state
        self.components: Dict[str, ComponentHealth] = {}
        self.restart_counts: Dict[str, int] = {}
        self.last_restart_time: Dict[str, float] = {}
        
        # Cameras config
        self.cameras = self.config['intersection']['cameras']
        
        # Services to monitor
        self.services = self.health_config.auto_restart_services or [
            'isfahan-deepstream',
            'isfahan-triton',
            'isfahan-mqtt-bridge',
        ]
        
        # Running state
        self.running = False
        self.start_time = time.time()
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _parse_health_config(self) -> HealthConfig:
        return HealthConfig(
            check_interval=int(os.getenv('CHECK_INTERVAL', '30')),
            gpu_memory_threshold=int(os.getenv('GPU_MEMORY_THRESHOLD', '90')),
            gpu_temp_threshold=int(os.getenv('GPU_TEMP_THRESHOLD', '85')),
            gpu_power_threshold=int(os.getenv('GPU_POWER_THRESHOLD', '250')),
            camera_timeout=int(os.getenv('CAMERA_TIMEOUT', '10')),
            camera_fps_min=int(os.getenv('CAMERA_FPS_MIN', '20')),
            mqtt_heartbeat_timeout=int(os.getenv('MQTT_HEARTBEAT_TIMEOUT', '60')),
            triton_health_endpoint=os.getenv('TRITON_HEALTH_ENDPOINT', 'http://localhost:8000/v2/health/ready'),
            deepstream_health_endpoint=os.getenv('DEEPSTREAM_HEALTH_ENDPOINT', 'http://localhost:8080/health'),
            alert_webhook=os.getenv('ALERT_WEBHOOK'),
            alert_webhook_timeout=int(os.getenv('ALERT_WEBHOOK_TIMEOUT', '10')),
            auto_restart_services=os.getenv('AUTO_RESTART_SERVICES', '').split(',') if os.getenv('AUTO_RESTART_SERVICES') else None,
            max_restart_attempts=int(os.getenv('MAX_RESTART_ATTEMPTS', '3')),
            restart_cooldown=int(os.getenv('RESTART_COOLDOWN', '300')),
            metrics_port=int(os.getenv('METRICS_PORT', '9090')),
        )
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def start(self):
        """شروع مانیتورینگ"""
        self.running = True
        
        # Start Prometheus metrics server
        start_http_server(self.health_config.metrics_port)
        logger.info(f"Prometheus metrics server started on port {self.health_config.metrics_port}")
        
        # Start web server for health endpoint
        app = web.Application()
        app.router.add_get('/health', self._health_endpoint)
        app.router.add_get('/metrics', self._metrics_endpoint)
        app.router.add_get('/components', self._components_endpoint)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 9091)
        await site.start()
        logger.info("Health HTTP server started on port 9091")
        
        # Start monitoring loops
        tasks = [
            asyncio.create_task(self._gpu_monitor_loop()),
            asyncio.create_task(self._camera_monitor_loop()),
            asyncio.create_task(self._container_monitor_loop()),
            asyncio.create_task(self._mqtt_monitor_loop()),
            asyncio.create_task(self._service_health_loop()),
            asyncio.create_task(self._system_monitor_loop()),
        ]
        
        logger.info("🏥 Health Monitor started")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await runner.cleanup()
    
    async def _health_endpoint(self, request):
        """Health check endpoint for load balancers"""
        overall_status = "healthy"
        for comp in self.components.values():
            if comp.status == "critical":
                overall_status = "critical"
                break
            elif comp.status == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return web.json_response({
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'uptime_seconds': time.time() - self.start_time,
            'components': {name: asdict(comp) for name, comp in self.components.items()},
        }, status=200 if overall_status != "critical" else 503)
    
    async def _metrics_endpoint(self, request):
        """Prometheus metrics endpoint"""
        # Metrics are exposed via prometheus_client on separate port
        return web.Response(text="Metrics available on port 9090")
    
    async def _components_endpoint(self, request):
        """Detailed component status"""
        return web.json_response({
            'components': {name: asdict(comp) for name, comp in self.components.items()},
        })
    
    async def _gpu_monitor_loop(self):
        """GPU monitoring loop"""
        while self.running:
            try:
                if self.nvml_available:
                    await self._check_gpu_health()
                else:
                    self._update_component('gpu', 'unknown', {'error': 'NVML not available'})
            except Exception as e:
                logger.error(f"GPU monitor error: {e}")
                self._update_component('gpu', 'critical', {'error': str(e)})
            
            await asyncio.sleep(self.health_config.check_interval)
    
    async def _check_gpu_health(self):
        """بررسی سلامت GPU"""
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_pct = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            mem_util = util.memory
            
            # Update Prometheus
            GPU_MEMORY_USAGE.labels(gpu=f'gpu{i}').set(mem_info.used)
            GPU_TEMPERATURE.labels(gpu=f'gpu{i}').set(temp)
            GPU_POWER.labels(gpu=f'gpu{i}').set(power)
            
            # Determine status
            status = "healthy"
            issues = []
            
            if mem_used_pct > self.health_config.gpu_memory_threshold:
                status = "critical"
                issues.append(f"Memory {mem_used_pct:.1f}% > {self.health_config.gpu_memory_threshold}%")
            
            if temp > self.health_config.gpu_temp_threshold:
                status = "critical" if status != "critical" else "critical"
                issues.append(f"Temperature {temp}°C > {self.health_config.gpu_temp_threshold}°C")
            
            if power > self.health_config.gpu_power_threshold:
                status = "degraded" if status == "healthy" else status
                issues.append(f"Power {power:.1f}W > {self.health_config.gpu_power_threshold}W")
            
            details = {
                'name': name,
                'memory_used_mb': mem_info.used // 1024 // 1024,
                'memory_total_mb': mem_info.total // 1024 // 1024,
                'memory_percent': round(mem_used_pct, 1),
                'temperature_c': temp,
                'power_w': round(power, 1),
                'gpu_util_percent': gpu_util,
                'memory_util_percent': mem_util,
                'issues': issues,
            }
            
            self._update_component(f'gpu_{i}', status, details)
            HEALTH_CHECKS.labels(component=f'gpu_{i}', status=status).inc()
    
    async def _camera_monitor_loop(self):
        """Camera monitoring loop"""
        while self.running:
            try:
                await self._check_cameras()
            except Exception as e:
                logger.error(f"Camera monitor error: {e}")
            
            await asyncio.sleep(self.health_config.check_interval)
    
    async def _check_cameras(self):
        """بررسی وضعیت دوربین‌ها"""
        for cam in self.cameras:
            cam_id = cam['id']
            rtsp_url = cam['rtsp']
            
            # Test RTSP connection
            start = time.time()
            try:
                # Quick test with ffprobe (non-blocking)
                proc = await asyncio.create_subprocess_exec(
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,r_frame_rate',
                    '-of', 'json', rtsp_url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.health_config.camera_timeout)
                
                latency = (time.time() - start) * 1000
                
                if proc.returncode == 0:
                    info = json.loads(stdout.decode())
                    stream = info['streams'][0]
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    
                    # Parse frame rate
                    fps_str = stream.get('r_frame_rate', '0/1')
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den > 0 else 0
                    
                    status = "healthy"
                    issues = []
                    
                    if fps < self.health_config.camera_fps_min:
                        status = "degraded"
                        issues.append(f"FPS {fps:.1f} < {self.health_config.camera_fps_min}")
                    
                    if latency > 5000:
                        status = "degraded" if status == "healthy" else status
                        issues.append(f"High latency {latency:.0f}ms")
                    
                    details = {
                        'rtsp_url': rtsp_url,
                        'resolution': f"{width}x{height}",
                        'fps': round(fps, 1),
                        'latency_ms': round(latency, 0),
                        'position': cam.get('position', 'unknown'),
                        'issues': issues,
                    }
                    
                    CAMERA_FPS.labels(camera_id=cam_id).set(fps)
                    CAMERA_LATENCY.labels(camera_id=cam_id).set(latency)
                    
                else:
                    status = "critical"
                    details = {
                        'rtsp_url': rtsp_url,
                        'error': stderr.decode()[:200],
                        'issues': ['Connection failed'],
                    }
                    CAMERA_FPS.labels(camera_id=cam_id).set(0)
                
            except asyncio.TimeoutError:
                status = "critical"
                details = {
                    'rtsp_url': rtsp_url,
                    'error': 'Timeout',
                    'issues': [f'Timeout > {self.health_config.camera_timeout}s'],
                }
                CAMERA_FPS.labels(camera_id=cam_id).set(0)
            
            except Exception as e:
                status = "critical"
                details = {
                    'rtsp_url': rtsp_url,
                    'error': str(e),
                    'issues': ['Exception'],
                }
                CAMERA_FPS.labels(camera_id=cam_id).set(0)
            
            self._update_component(f'camera_{cam_id}', status, details)
            HEALTH_CHECKS.labels(component=f'camera_{cam_id}', status=status).inc()
    
    async def _container_monitor_loop(self):
        """Container monitoring loop"""
        while self.running:
            try:
                await self._check_containers()
            except Exception as e:
                logger.error(f"Container monitor error: {e}")
            
            await asyncio.sleep(self.health_config.check_interval)
    
    async def _check_containers(self):
        """بررسی وضعیت کانتینرها"""
        for service_name in self.services:
            try:
                container = self.docker_client.containers.get(service_name)
                state = container.status
                health = container.attrs.get('State', {}).get('Health', {}).get('Status', 'none')
                
                if state == 'running':
                    if health == 'healthy':
                        status = "healthy"
                    elif health == 'unhealthy':
                        status = "critical"
                    elif health == 'starting':
                        status = "degraded"
                    else:
                        status = "healthy"  # No health check defined
                else:
                    status = "critical"
                
                details = {
                    'state': state,
                    'health': health,
                    'restart_count': container.attrs.get('RestartCount', 0),
                    'started_at': container.attrs.get('State', {}).get('StartedAt', ''),
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                }
                
                CONTAINER_STATUS.labels(container=service_name).set(1 if state == 'running' else 0)
                
            except docker.errors.NotFound:
                status = "critical"
                details = {'error': 'Container not found'}
                CONTAINER_STATUS.labels(container=service_name).set(0)
            except Exception as e:
                status = "critical"
                details = {'error': str(e)}
                CONTAINER_STATUS.labels(container=service_name).set(0)
            
            self._update_component(f'container_{service_name}', status, details)
            HEALTH_CHECKS.labels(component=f'container_{service_name}', status=status).inc()
            
            # Auto-restart if critical
            if status == "critical":
                await self._maybe_restart_container(service_name)
    
    async def _mqtt_monitor_loop(self):
        """MQTT monitoring loop"""
        while self.running:
            try:
                await self._check_mqtt()
            except Exception as e:
                logger.error(f"MQTT monitor error: {e}")
            
            await asyncio.sleep(self.health_config.check_interval)
    
    async def _check_mqtt(self):
        """بررسی اتصال MQTT"""
        # Check local MQTT (DeepStream)
        local_status = await self._check_mqtt_broker('localhost', 1883, 'local')
        self._update_component('mqtt_local', local_status['status'], local_status)
        HEALTH_CHECKS.labels(component='mqtt_local', status=local_status['status']).inc()
        MQTT_CONNECTED.labels(broker='local').set(1 if local_status['status'] == 'healthy' else 0)
        
        # Check remote MQTT (UTC/SCADA)
        remote_status = await self._check_mqtt_broker(
            self.config['intersection']['integration']['mqtt_broker'].split(':')[0],
            int(self.config['intersection']['integration']['mqtt_broker'].split(':')[1]) if ':' in self.config['intersection']['integration']['mqtt_broker'] else 8883,
            'remote'
        )
        self._update_component('mqtt_remote', remote_status['status'], remote_status)
        HEALTH_CHECKS.labels(component='mqtt_remote', status=remote_status['status']).inc()
        MQTT_CONNECTED.labels(broker='remote').set(1 if remote_status['status'] == 'healthy' else 0)
    
    async def _check_mqtt_broker(self, host: str, port: int, broker_type: str) -> Dict:
        """بررسی اتصال به بروکر MQTT"""
        try:
            import paho.mqtt.client as mqtt
            
            client = mqtt.Client(client_id=f"health-check-{broker_type}-{os.getpid()}")
            client.connect(host, port, keepalive=5)
            client.loop_start()
            
            # Wait for connection
            await asyncio.sleep(1)
            
            connected = client.is_connected()
            client.loop_stop()
            client.disconnect()
            
            if connected:
                return {'status': 'healthy', 'host': host, 'port': port, 'latency_ms': 0}
            else:
                return {'status': 'critical', 'host': host, 'port': port, 'error': 'Not connected'}
                
        except Exception as e:
            return {'status': 'critical', 'host': host, 'port': port, 'error': str(e)}
    
    async def _service_health_loop(self):
        """Service health endpoints monitoring"""
        while self.running:
            try:
                # Check Triton
                triton_status = await self._check_http_endpoint(
                    self.health_config.triton_health_endpoint, 'triton'
                )
                self._update_component('triton', triton_status['status'], triton_status)
                HEALTH_CHECKS.labels(component='triton', status=triton_status['status']).inc()
                
                # Check DeepStream
                ds_status = await self._check_http_endpoint(
                    self.health_config.deepstream_health_endpoint, 'deepstream'
                )
                self._update_component('deepstream', ds_status['status'], ds_status)
                HEALTH_CHECKS.labels(component='deepstream', status=ds_status['status']).inc()
                
            except Exception as e:
                logger.error(f"Service health check error: {e}")
            
            await asyncio.sleep(self.health_config.check_interval)
    
    async def _check_http_endpoint(self, url: str, name: str) -> Dict:
        """بررسی endpoint HTTP"""
        try:
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    latency = (time.time() - start) * 1000
                    if resp.status == 200:
                        data = await resp.json() if 'json' in resp.content_type else {}
                        return {'status': 'healthy', 'url': url, 'latency_ms': round(latency, 0), 'data': data}
                    else:
                        return {'status': 'degraded', 'url': url, 'latency_ms': round(latency, 0), 'http_status': resp.status}
        except asyncio.TimeoutError:
            return {'status': 'critical', 'url': url, 'error': 'Timeout'}
        except Exception as e:
            return {'status': 'critical', 'url': url, 'error': str(e)}
    
    async def _system_monitor_loop(self):
        """System resources monitoring"""
        while self.running:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory
                mem = psutil.virtual_memory()
                mem_percent = mem.percent
                
                # Disk
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                
                # Network
                net = psutil.net_io_counters()
                
                # Load average
                load1, load5, load15 = psutil.getloadavg()
                
                status = "healthy"
                issues = []
                
                if cpu_percent > 90:
                    status = "degraded"
                    issues.append(f"CPU {cpu_percent:.1f}%")
                
                if mem_percent > 90:
                    status = "critical" if status != "critical" else "critical"
                    issues.append(f"Memory {mem_percent:.1f}%")
                
                if disk_percent > 90:
                    status = "critical" if status != "critical" else "critical"
                    issues.append(f"Disk {disk_percent:.1f}%")
                
                details = {
                    'cpu_percent': round(cpu_percent, 1),
                    'memory_percent': round(mem_percent, 1),
                    'memory_available_gb': round(mem.available / 1024**3, 1),
                    'disk_percent': round(disk_percent, 1),
                    'disk_free_gb': round(disk.free / 1024**3, 1),
                    'load_average': [round(load1, 2), round(load5, 2), round(load15, 2)],
                    'network_bytes_sent': net.bytes_sent,
                    'network_bytes_recv': net.bytes_recv,
                    'issues': issues,
                }
                
                self._update_component('system', status, details)
                HEALTH_CHECKS.labels(component='system', status=status).inc()
                
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                self._update_component('system', 'critical', {'error': str(e)})
            
            await asyncio.sleep(self.health_config.check_interval * 2)  # Less frequent
    
    def _update_component(self, name: str, status: str, details: Dict):
        """به‌روزرسانی وضعیت کامپوننت"""
        now = time.time()
        
        if name in self.components:
            comp = self.components[name]
            comp.last_check = now
            comp.status = status
            comp.details = details
            if status in ('critical', 'degraded'):
                comp.consecutive_failures += 1
            else:
                comp.consecutive_failures = 0
        else:
            self.components[name] = ComponentHealth(
                name=name,
                status=status,
                last_check=now,
                details=details,
                consecutive_failures=1 if status in ('critical', 'degraded') else 0,
            )
    
    async def _maybe_restart_container(self, service_name: str):
        """تلاش برای ریستارت خودکار کانتینر"""
        now = time.time()
        
        # Check cooldown
        if service_name in self.last_restart_time:
            if now - self.last_restart_time[service_name] < self.health_config.restart_cooldown:
                return
        
        # Check max attempts
        attempts = self.restart_counts.get(service_name, 0)
        if attempts >= self.health_config.max_restart_attempts:
            logger.warning(f"Max restart attempts reached for {service_name}")
            await self._send_alert('critical', 'max_restarts_reached', {
                'service': service_name,
                'attempts': attempts,
            })
            return
        
        # Attempt restart
        logger.warning(f"Attempting auto-restart of {service_name} (attempt {attempts + 1})")
        
        try:
            container = self.docker_client.containers.get(service_name)
            container.restart(timeout=30)
            
            self.restart_counts[service_name] = attempts + 1
            self.last_restart_time[service_name] = now
            
            AUTO_RESTARTS.labels(container=service_name).inc()
            
            await self._send_alert('warning', 'container_restarted', {
                'service': service_name,
                'attempt': attempts + 1,
                'max_attempts': self.health_config.max_restart_attempts,
            })
            
            logger.info(f"✅ Restarted {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to restart {service_name}: {e}")
            await self._send_alert('critical', 'restart_failed', {
                'service': service_name,
                'error': str(e),
            })
    
    async def _send_alert(self, severity: str, alert_type: str, details: Dict):
        """ارسال هشدار به وبهوک"""
        ALERTS_SENT.labels(severity=severity, type=alert_type).inc()
        
        alert = {
            'alert_id': f"alert_{int(time.time())}_{alert_type}",
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'severity': severity,
            'type': alert_type,
            'intersection': 'chaharbag_azadi',
            'details': details,
        }
        
        logger.warning(f"🚨 ALERT [{severity.upper()}] {alert_type}: {details}")
        
        # Send to webhook if configured
        if self.health_config.alert_webhook:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.health_config.alert_webhook,
                        json=alert,
                        timeout=aiohttp.ClientTimeout(total=self.health_config.alert_webhook_timeout)
                    )
            except Exception as e:
                logger.error(f"Failed to send alert webhook: {e}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Isfahan Health Monitor')
    parser.add_argument('--config', default='/app/configs/intersection.yaml', help='Config file')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    monitor = IsfahanHealthMonitor(config_path=args.config)
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Monitor error: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())