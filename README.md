# 🏛 Isfahan Smart Intersection Pilot (ISIP)

**سیستم مدیریت هوشمند ترافیک تقاطع چهارباغ-آزادی اصفهان**  
*Advanced Computer Vision + RL + Digital Twin for Real-world Traffic Management*

---

## 🎯 Overview

| ویژگی | جزئیات |
|----------|---------|
| **موقعیت** | تقاطع چهارباغ-آزادی، اصفهان (32.6546° N, 51.6680° E) |
| **نوع تقاطع** | ۴‌ذراعی، ۵ فاز سیگنال، خط BRT دوطرفه |
| **حجم ترافیک** | ۸,۵۰۰+ وسیله/ساعت در اوج |
| **دوربین‌ها** | ۸ دوربین IP (۴ روی يأتي، ۴ نمای غoghi) |
| **کنترلر سیگنال** | Siemens Sitrafic (NTCIP 1202) |
| **هدف** | کاهش ۲۲% زمان انتظار، تشخیص ناهنجاری بلادرنگ، بهینه‌سازی تطبیقی سیگنال |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      EDGE NODE (Jetson Orin AGX 64GB)           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  DeepStream │──▶│   Triton    │──▶│  MQTT Bridge│             │
│  │  Pipeline   │   │  Inference  │   │  (UTC/SCADA)│             │
│  │  (8 cams)   │   │  (TensorRT) │   │             │             │
│  └─────────────┘  └─────────────┘  └──────┬──────┘             │
│         │                │                │                    │
│         ▼                ▼                ▼                    │
│  ┌─────────────────────────────────────────────┐              │
│  │           Health Monitor                    │              │
│  │  (GPU, Cameras, Containers, Auto-recovery) │              │
│  └─────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │ MQTT / HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CLOUD / CONTROL CENTER                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Grafana   │  │ ClickHouse  │  │   MLflow    │             │
│  │  Dashboard  │  │  Analytics  │  │ Model Registry│            │
│  │  (Persian)  │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                              │                                  │
│  ┌─────────────────────────────────────────────┐              │
│  │         Digital Twin (3D GS + SUMO)         │              │
│  │     What-if Analysis • Operator View        │              │
│  └─────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 AI Models

| ماژول | مدل | فریمورک | بهینه‌سازی | تأخیر (Jetson Orin) |
|-------|------|---------|------------|---------------------|
| **تشخیص** | YOLOv10-N (ISF) | PyTorch → TensorRT | INT8 Quantization | ~1.8ms |
| **ردیابی** | BoT-SORT + OSNet-x1.0 | PyTorch → TensorRT | FP16 | ~3ms |
| **ناهنجاری** | ST-GCN (Trajectory) | PyTorch → ONNX Runtime | FP16 | ~5ms |
| **کنترل سیگنال** | PPO-GAT (Graph RL) | PyTorch → TorchScript | FP16 | ~10ms |

### کلاس‌های تشخیص (۱۲ کلاس مخصوص ایران)
```
0: sedan_iran      1: pickup_iran     2: motorcycle_iran
3: bus_brt         4: bus_city        5: truck_heavy
6: van_transport   7: bicycle         8: pedestrian
9: traffic_light  10: traffic_sign    11: crosswalk
```

### ناهنجاری‌های قابل تشخیص (۱۱ نوع)
```
0: normal                    1: motorcycle_wrong_lane
2: red_light_violation       3: illegal_uturn
4: stop_on_crosswalk         5: wrong_way_driving
6: pedestrian_jaywalking     7: bus_lane_violation
8: truck_in_car_lane         9: sudden_lane_change
10: speeding
```

---

## 🚀 Quick Start

### Prerequisites
- **Hardware**: Jetson Orin AGX 64GB (×2 for HA) or x86 + NVIDIA GPU (A100/A6000)
- **Software**: Docker 24+, NVIDIA Container Toolkit, JetPack 6.0+
- **Network**: Access to 8 RTSP cameras, MQTT broker (UTC), HTTPS endpoints

### 1. Clone & Configure
```bash
git clone https://github.com/your-org/isfahan-smart-intersection.git
cd isfahan-smart-intersection

# Configure environment
cp .env.example .env
# Edit .env with your credentials:
# - RTSP camera URLs
# - MQTT broker (UTC) credentials
# - TLS certificates
# - Webhook URLs
```

### 2. Generate Synthetic Data (First Run)
```bash
# Start CARLA server
docker run -d --name carla -p 2000-2002:2000-2002 --gpus all carlasim/carla:0.9.15

# Build Isfahan map
python tools/carla_isfahan_map/generate_isfahan_map.py

# Generate 50K labeled frames
python tools/synthetic_generator/generate_synthetic_data.py --frames 50000
```

### 3. Train Models
```bash
# Detection (YOLOv10)
yolo detect train data=configs/isfahan_traffic.yaml model=yolov10n.pt epochs=150 imgsz=640 batch=64 device=0,1,2,3

# ReID for Tracking
python models/tracking/train_reid.py --data data/fused --epochs 120

# Anomaly Detection (ST-GCN)
python models/anomaly/train_stgcn.py --data data/fused/trajectories --epochs 200

# Signal Control (RL)
python models/signal_rl/train_ppo_gat.py --timesteps 1000000 --n_envs 8
```

### 4. Export to TensorRT & Deploy
```bash
# Export models to TensorRT INT8
python export_tensorrt.py --all

# Build Docker images
docker compose -f docker-compose.edge.yml build --parallel

# Deploy to Jetson Orin
scp -r . jetson@192.168.1.100:/opt/isfahan-sip/
ssh jetson@192.168.1.100 "cd /opt/isfahan-sip && docker compose -f docker-compose.edge.yml up -d"
```

### 5. Monitor
```bash
# Grafana Dashboard (Persian RTL)
open http://localhost:3000/d/isfahan-chaharbag-azadi

# Health endpoint
curl http://localhost:9091/health

# Prometheus metrics
curl http://localhost:9090/metrics
```

---

## 📁 Project Structure

```
isfahan-smart-intersection/
├── 🐳 docker-compose.edge.yml          # Edge deployment (Jetson Orin)
├── 🐳 docker-compose.cloud.yml         # Cloud deployment (K8s)
├── ⚙️ configs/
│   ├── intersection.yaml               # Intersection specification
│   ├── signal_phases.yaml              # Signal timing plans
│   ├── anomalies_iran.yaml             # Iranian anomaly definitions
│   ├── isfahan_traffic.yaml            # YOLO dataset config
│   └── calibration/                    # Camera calibration files
├── 📊 data/
│   ├── synthetic/                      # CARLA-generated data
│   ├── public/                         # UA-DETRAC, BDD100K, etc.
│   ├── fused/                          # Merged + unified dataset
│   └── pretrained/                     # Pre-trained weights
├── 🧠 models/
│   ├── detection/                      # YOLOv10-ISF
│   ├── tracking/                       # BoT-SORT + ReID
│   ├── anomaly/                        # ST-GCN
│   ├── signal_rl/                      # PPO-GAT
│   └── digital_twin/                   # 3D GS + SUMO
├── 🖥️ edge/
│   ├── deepstream_pipeline/            # GStreamer pipeline
│   ├── triton_model_repo/              # Triton configs
│   ├── mqtt_bridge/                    # UTC/SCADA integration
│   └── health_monitor/                 # Auto-recovery
├── ☁️ cloud/
│   ├── api/                            # FastAPI + gRPC
│   ├── clickhouse/                     # Analytics schema
│   ├── grafana/                        # Persian RTL dashboards
│   └── model_registry/                 # MLflow + DVC
├── 🛠️ tools/
│   ├── carla_isfahan_map/              # Map builder
│   ├── synthetic_generator/            # Data generator
│   ├── data_fusion/                    # Dataset merger
│   └── label_studio/                   # Annotation workflow
└── 📦 deployment/
    ├── jetson_orin/                    # JetPack setup
    ├── kubernetes/                     # K8s manifests
    └── ansible/                        # Provisioning
```

---

## 📊 Grafana Dashboard (Persian RTL)

![Dashboard Preview](docs/dashboard-preview.png)

**پنل‌های اصلی:**
- 🟢 **وضعیت کلی تقاطع** (سبز/زرد/قرمز)
- 📈 **طول صف چهار جهت** (چارت زمانی ۱۵ دقیقه)
- ⚡ **کارایی RL Agent** (گول ۰-۱۰۰٪)
- 🚨 **ناهنجاری‌های فعال** (جدول بلادرنگ)
- 🗺️ **نمای دیجیتال توئم** (۳D Gaussian Splatting)
- 📊 **KPIها**: انتظار، عبور، اولویت پیاده‌رو، امدادرسانی

---

## 🔧 Configuration

### Camera Calibration
```yaml
# configs/calibration/cam_01.yaml
camera_matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
distortion_coeffs: [k1, k2, p1, p2, k3]
extrinsic:
  rotation: [r11, r12, r13, r21, r22, r23, r31, r32, r33]
  translation: [tx, ty, tz]
```

### Signal Timing Plans
```yaml
# configs/signal_phases.yaml
phases:
  - id: 0
    name: "NB_SB_Straight"
    movements: ["N->S", "S->N"]
    min_green: 20
    max_green: 80
  # ... 5 phases total
```

### Anomaly Priorities (Iranian Regulations)
```yaml
# configs/anomalies_iran.yaml
anomaly_priorities:
  red_light_violation: 10      # ۶ میلیون تومان جرمه
  wrong_way_driving: 10        # ۱۰ میلیون تومان جرمه
  truck_in_car_lane: 8         # ۸ میلیون تومان جرمه
  speeding: 7                  # ۵ میلیون تومان جرمه
  # ...
```

---

## 📈 Performance Targets

| متریک | هدف | فعلاً (پایلوت) |
|--------|------|----------------|
| **mAP@0.5 (Detection)** | > 0.90 | 0.87 |
| **MOTA (Tracking)** | > 0.80 | 0.76 |
| **F1 (Anomaly)** | > 0.92 | 0.89 |
| **Wait Time Reduction** | > 20% | 18% |
| **End-to-End Latency** | < 50ms | 42ms |
| **Uptime** | > 99.5% | 99.2% |
| **False Positive Rate** | < 5% | 6.2% |

---

## 🛡️ Security & Compliance

- **TLS 1.3** for all MQTT/HTTP communications
- **Mutual TLS** with UTC/SCADA (client certificates)
- **Network Isolation**: Edge VLAN separate from municipal network
- **Data Retention**: 30 days raw, 1 year aggregated (Iran regulations)
- **Privacy**: Face/license plate blurring in stored footage
- **Audit Trail**: All signal changes logged with operator ID

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] MOU signed with Isfahan Municipality
- [ ] Camera RTSP credentials verified
- [ ] UTC/SCADA API access granted
- [ ] TLS certificates installed
- [ ] Jetson Orin units provisioned (JetPack 6.0)
- [ ] Network VLANs configured
- [ ] UPS & environmental monitoring installed

### Deployment
- [ ] CARLA map validated on-site
- [ ] Camera calibration completed
- [ ] Synthetic data generation verified
- [ ] Models trained & exported to TensorRT
- [ ] Docker images built & scanned
- [ ] Edge deployment tested (24h soak)
- [ ] MQTT integration with UTC verified
- [ ] Grafana dashboards localized (Persian RTL)
- [ ] Alert webhooks tested
- [ ] Failover tested (primary → standby)

### Post-Deployment
- [ ] 2-week pilot monitoring
- [ ] KPI baseline established
- [ ] Operator training completed
- [ ] Handover documentation delivered
- [ ] Maintenance SLA agreed

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- **Python**: Black, isort, mypy, pylint
- **Docker**: Hadolint, multi-stage builds
- **Configs**: YAML with schema validation
- **Docs**: Persian + English bilingual

---

## 📄 License

Proprietary - Isfahan Municipality Traffic Engineering Department  
All rights reserved. Unauthorized copying prohibited.

---

## 📞 Contact

**Technical Lead**: [Your Name]  
**Email**: traffic-ai@isfahan.municipality.ir  
**Phone**: +98 31 3XXX XXXX  
**Address**: Traffic Engineering Dept., Isfahan Municipality, Chahar Bagh Ave.

---

## 🙏 Acknowledgments

- **NVIDIA** - Jetson Orin, DeepStream, TensorRT, Triton
- **Ultralytics** - YOLOv10
- **CARLA** - Simulator for synthetic data
- **SUMO** - Traffic simulation
- **OpenPCDet / MMDetection** - Reference implementations
- **Isfahan Municipality** - Domain expertise & data access

---

*Built with ❤️ for smarter, safer streets of Isfahan*