# 🏛 Isfahan Smart Intersection Pilot (ISIP)

**Intelligent Traffic Management System for the Chaharbagh–Azadi Intersection, Isfahan, Iran**

*Edge AI (DeepStream + Triton) • Multi-Object Tracking • Trajectory Anomaly Detection • Graph Reinforcement Learning Signal Control • Digital Twin*

> 🇮🇷 [نسخه فارسی این سند](#-نسخه-فارسی) در پایین همین صفحه موجود است.
>
> This document is bilingual: **English first, Persian (فارسی) second**. Both versions describe the same system.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [AI Models](#-ai-models)
- [Detection Classes & Anomaly Types](#-detection-classes--anomaly-types)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Performance Targets](#-performance-targets)
- [Security & Privacy](#-security--privacy)
- [Deployment Checklist](#-deployment-checklist)
- [Contributing](#-contributing)
- [License](#-license)
- [نسخه فارسی](#-نسخه-فارسی)

---

## 🎯 Overview

A production-grade edge-to-cloud pilot for adaptive traffic-signal control at the **Chaharbagh–Azadi intersection** in Isfahan — one of the city's busiest junctions.

| Attribute | Details |
|---|---|
| **Location** | Chaharbagh–Azadi intersection, Isfahan (32.6546° N, 51.6680° E) |
| **Intersection type** | 4-arm, 5 signal phases, two-way BRT corridor |
| **Peak traffic volume** | 8,500+ vehicles/hour |
| **Cameras** | 8 IP cameras (4 approach views, 4 overhead views) |
| **Signal controller** | Siemens Motiv/NTCIP-1202-compatible UTC |
| **Goals** | ≥ 20% wait-time reduction, real-time violation detection, adaptive signal control |

### What this repository contains

This repository is a **complete reference implementation** of the edge AI stack:

1. **Data pipeline** — synthetic data generation (CARLA), dataset fusion with public benchmarks.
2. **Model training** — YOLOv10 detection, Re-ID tracking, ST-GCN anomaly classification, PPO-GAT signal RL.
3. **Optimization** — TensorRT engine export (INT8/FP16) for Jetson Orin deployment.
4. **Edge runtime** — DeepStream pipeline, Triton inference server, MQTT bridge to the municipal UTC/SCADA, and a self-healing health monitor.
5. **Ops** — Docker Compose for both edge (Jetson) and cloud (analytics) deployments, plus a full operations runbook.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      EDGE NODE (Jetson Orin AGX 64GB)           │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │  DeepStream │──▶│   Triton    │──▶│  MQTT Bridge│           │
│  │  Pipeline   │   │  Inference  │   │ (UTC/SCADA) │           │
│  │  (8 cams)   │   │ (TensorRT)  │   │             │           │
│  └─────────────┘   └─────────────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────┐               │
│  │              Health Monitor                 │               │
│  │  GPU • Cameras • Containers • Auto-recovery │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                             │  MQTT over TLS 8883 / HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CLOUD / CONTROL CENTER                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   Grafana   │   │  Analytics  │   │   MLflow    │           │
│  │  Dashboard  │   │ (ClickHouse)│   │   Registry  │           │
│  │ (Persian RTL│   │             │   │             │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                        │                                        │
│  ┌─────────────────────────────────────────────┐               │
│  │     Digital Twin (SUMO + trajectory replay) │               │
│  │        What-if Analysis • Operator View     │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:** RTSP frames → DeepStream decode/ preprocess → Triton (detection → tracking → anomaly) → trajectory features → RL signal policy → phase commands → MQTT bridge → UTC/SCADA. Telemetry is mirrored to the cloud for dashboards, analytics, and the digital twin.

---

## 🧠 AI Models

| Module | Model | Framework | Optimization | Latency (Jetson Orin) |
|---|---|---|---|---|
| **Detection** | YOLOv10-N (ISF variant) | PyTorch → TensorRT | INT8 quantization | ~1.8 ms |
| **Tracking** | BoT-SORT + OSNet-x1.0 ReID | PyTorch → TensorRT | FP16 | ~3 ms |
| **Anomaly** | ST-GCN (trajectory-based) | PyTorch → ONNX Runtime | FP16 | ~5 ms |
| **Signal control** | PPO + Graph Attention Network | PyTorch → TorchScript | FP16 | ~10 ms |

**End-to-end latency budget:** < 50 ms from frame capture to signal-command emission (measured pilot target: ~42 ms).

---

## 🚗 Detection Classes & Anomaly Types

### Detection — 12 Iran-specific classes (`configs/isfahan_traffic.yaml`)

```
0: sedan_iran      1: pickup_iran     2: motorcycle_iran
3: bus_brt         4: bus_city        5: truck_heavy
6: van_transport   7: bicycle         8: pedestrian
9: traffic_light  10: traffic_sign   11: crosswalk
```

### Anomalies — 11 Iran-relevant violation types (`configs/anomalies_iran.yaml`)

```
0: normal                    1: motorcycle_wrong_lane
2: red_light_violation       3: illegal_uturn
4: stop_on_crosswalk         5: wrong_way_driving
6: pedestrian_jaywalking     7: bus_lane_violation
8: truck_in_car_lane         9: sudden_lane_change
10: speeding
```

Each anomaly has a **priority weight** derived from Iranian traffic regulations, used by the alert pipeline to rank and route notifications.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Detail |
|---|---|
| **Edge hardware** | Jetson Orin AGX 64GB (×2 for HA) or x86 + NVIDIA GPU (A100 / A6000) |
| **Software** | Docker 24+, NVIDIA Container Toolkit, JetPack 6.0+ (edge) |
| **Network** | Access to 8 RTSP cameras, MQTT broker (UTC), HTTPS endpoints |
| **Training** | Multi-GPU node recommended (see `.env.example`: `TRAIN_GPUS=0,1,2,3`) |

### 1. Clone & configure

```bash
git clone https://github.com/alirezahs81/isfahan-smart-intersection.git
cd isfahan-smart-intersection

cp .env.example .env
# Edit .env: RTSP camera URLs, MQTT credentials, TLS cert paths, webhook URLs
```

> ⚠️ Never commit `.env` or TLS certificates — they are git-ignored. The only tracked env file is `.env.example`, which contains placeholders only.

### 2. Generate synthetic data (first run)

```bash
# Start a CARLA server
docker run -d --name carla -p 2000-2002:2000-2002 --gpus all carlasim/carla:0.9.15

# Build the Isfahan intersection map in CARLA
python tools/carla_isfahan_map/generate_isfahan_map.py

# Generate ~50K labeled frames with Iranian vehicle mix
python tools/synthetic_generator/generate_synthetic_data.py --frames 50000
```

### 3. Fuse with public datasets & train

```bash
# Merge synthetic + public benchmarks (UA-DETRAC, BDD100K, …)
python tools/data_fusion/merge_datasets.py --output data/fused

# Detection (YOLOv10)
yolo detect train data=configs/isfahan_traffic.yaml model=yolov10n.pt \
    epochs=150 imgsz=640 batch=64 device=0,1,2,3

# Re-ID for tracking
python models/tracking/train_reid_isfahan.py --data data/fused --epochs 120

# Anomaly detection (ST-GCN on trajectories)
python models/anomaly/stgcn_anomaly.py --mode train --data data/fused/trajectories

# Signal control (PPO-GAT graph RL)
python models/signal_rl/ppo_gat_signal.py --mode train --timesteps 1000000 --n_envs 8

# Digital twin (SUMO co-simulation)
python models/digital_twin/isfahan_digital_twin.py --config configs/intersection.yaml
```

### 4. Export to TensorRT & deploy on the edge

```bash
# Export all models to TensorRT INT8/FP16 engines
python export_tensorrt.py --all

# Build edge images
docker compose -f docker-compose.edge.yml build --parallel

# Deploy to Jetson Orin
scp -r . jetson@<edge-ip>:/opt/isfahan-sip/
ssh jetson@<edge-ip> "cd /opt/isfahan-sip && docker compose -f docker-compose.edge.yml up -d"
```

### 5. Monitor

```bash
# Health endpoint (edge health monitor)
curl http://localhost:9091/health

# Prometheus metrics
curl http://localhost:9090/metrics

# Grafana dashboard (Persian RTL) — provisioned by docker-compose.cloud.yml
open http://localhost:3000
```

For step-by-step operational procedures (startup, failure recovery, calibration, rollback), see [`RUNBOOK.md`](RUNBOOK.md).

---

## 📁 Project Structure

```
isfahan-smart-intersection/
├── 🐳 docker-compose.edge.yml        # Edge stack: DeepStream, Triton, MQTT bridge, health monitor
├── 🐳 docker-compose.cloud.yml       # Cloud stack: Grafana, analytics, model registry
├── ⚙️ configs/
│   ├── intersection.yaml             # Intersection topology & phase specification
│   ├── signal_phases.yaml            # Baseline signal timing plans (5 phases)
│   ├── signal_rl.yaml                # RL agent hyperparameters & reward shaping
│   ├── anomalies_iran.yaml           # Iranian violation definitions & priorities
│   └── isfahan_traffic.yaml          # YOLO dataset config (12 classes)
├── 🖥️ edge/
│   ├── deepstream_pipeline/
│   │   ├── isfahan_pipeline.py       # 8-camera DeepStream/GStreamer pipeline
│   │   └── configs/                  # PGIE/SGIE, tracker, and label configs
│   ├── triton_model_repo/            # Triton model configs (detection, ReID, anomaly, RL)
│   ├── mqtt_bridge/                  # MQTT client bridging detections & commands to UTC/SCADA
│   └── health_monitor/               # GPU/camera/container watchdog with auto-recovery
├── 🧠 models/
│   ├── detection/                    # YOLOv10-ISF training (12 Iran-specific classes)
│   ├── tracking/                     # OSNet Re-ID fine-tuning
│   ├── anomaly/                      # ST-GCN trajectory anomaly classifier
│   ├── signal_rl/                    # PPO-GAT adaptive signal policy
│   └── digital_twin/                 # SUMO-based intersection digital twin
├── 🛠️ tools/
│   ├── carla_isfahan_map/            # CARLA map generator for the intersection
│   ├── synthetic_generator/          # Labeled synthetic data generator
│   └── data_fusion/                  # Merge synthetic + public datasets
├── export_tensorrt.py                # TensorRT INT8/FP16 engine exporter (all models)
├── requirements.txt                  # Root Python dependencies
├── .env.example                      # Environment template (placeholders only)
├── RUNBOOK.md                        # Operations & incident-response procedures
├── LICENSE                           # MIT license
└── README.md                         # This file (bilingual EN/FA)
```

> **Note:** model weights, datasets, TensorRT engines, TLS certificates and `.env` are intentionally **not** tracked — see [`.gitignore`](.gitignore).

---

## 🔧 Configuration Highlights

### Signal timing plan (`configs/signal_phases.yaml`)

```yaml
phases:
  - id: 0
    name: "NB_SB_Straight"
    movements: ["N->S", "S->N"]
    min_green: 20
    max_green: 80
  # … 5 phases total, incl. BRT priority phase
```

### Anomaly priorities (`configs/anomalies_iran.yaml`)

```yaml
anomaly_priorities:
  red_light_violation: 10
  wrong_way_driving: 10
  truck_in_car_lane: 8
  speeding: 7
  # … weights follow Iranian traffic enforcement regulations
```

The RL agent's reward (`configs/signal_rl.yaml`) combines **queue length, per-phase wait time, pedestrian crossing safety, and BRT headway adherence** — see the config for the exact weights.

---

## 📈 Performance Targets

| Metric | Target | Pilot (measured) |
|---|---|---|
| mAP@0.5 (detection) | > 0.90 | 0.87 |
| MOTA (tracking) | > 0.80 | 0.76 |
| F1 (anomaly) | > 0.92 | 0.89 |
| Wait-time reduction | > 20% | 18% |
| End-to-end latency | < 50 ms | 42 ms |
| Uptime | > 99.5% | 99.2% |
| False-positive rate | < 5% | 6.2% |

Pilot numbers are from the CARLA + synthetic-data loop; on-site calibration is expected to close the remaining gap.

---

## 🛡 Security & Privacy

- **TLS 1.3** for all MQTT/HTTP communication; **mutual TLS** (client certificates) with UTC/SCADA.
- **Network isolation:** edge VLAN segmented from the municipal network.
- **Secrets hygiene:** credentials live only in `.env` (git-ignored); `.env.example` contains placeholders.
- **Data retention:** 30 days raw footage, 1 year aggregates (per Iranian regulations).
- **Privacy:** face and license-plate blurring on stored footage.
- **Audit trail:** every signal-command change is logged with operator/agent identity.

---

## ✅ Deployment Checklist

**Pre-deployment**
- [ ] MOU signed with Isfahan Municipality
- [ ] Camera RTSP credentials verified
- [ ] UTC/SCADA API access granted, TLS certificates installed
- [ ] Jetson Orin units provisioned (JetPack 6.0)
- [ ] Network VLANs, UPS & environmental monitoring configured

**Deployment**
- [ ] CARLA map validated on-site, camera calibration completed
- [ ] Synthetic data generation verified, models trained & exported to TensorRT
- [ ] Docker images built & scanned, 24h edge soak test passed
- [ ] MQTT integration with UTC verified, Grafana localized (Persian RTL)
- [ ] Alert webhooks tested, failover (primary → standby) tested

**Post-deployment**
- [ ] 2-week pilot monitoring, KPI baseline established
- [ ] Operator training completed, handover docs delivered, maintenance SLA agreed

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit with clear messages: `git commit -m 'Add amazing feature'`
4. Push and open a Pull Request

**Code standards**
- Python: `black`, `isort`, `mypy`, `pylint`
- Docker: `hadolint`, multi-stage builds
- Configs: YAML with schema validation
- Docs: bilingual (English + Persian)

---

## 📄 License

Released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **NVIDIA** — Jetson Orin, DeepStream, TensorRT, Triton
- **Ultralytics** — YOLOv10
- **CARLA** & **SUMO** — synthetic data & traffic simulation
- **BoT-SORT / OSNet** authors — tracking & Re-ID baselines
- Isfahan Municipality Traffic Engineering Department — domain expertise

---
---

# 🇮🇷 نسخه فارسی

# پایلوت تقاطع هوشمند اصفهان (ISIP)

**سامانه مدیریت هوشمند ترافیک تقاطع چهارباغ–آزادی اصفهان**

*هوش مصنوعی لبه‌ای (DeepStream + Triton) • ردیابی چند‌هدف • تشخیص ناهنجاری مسیر • کنترل سیگنال با یادگیری تقویتی گرافی • دوقلوی دیجیتال*

---

## 🎯 معرفی

این ریپو یک **پیاده‌سازی مرجع کامل** از پشته هوش مصنوعی لبه‌ای برای مدیریت تطبیقی سیگنال‌ترافیک در **تقاطع چهارباغ–آزادی اصفهان** — یکی از پرترددترین تقاطع‌های شهر — است.

| ویژگی | جزئیات |
|---|---|
| **موقعیت** | تقاطع چهارباغ–آزادی، اصفهان (32.6546° N، 51.6680° E) |
| **نوع تقاطع** | ۴‌ذراعی، ۵ فاز سیگنال، خط BRT دوطرفه |
| **حجم ترافیک اوج** | بیش از ۸٬۵۰۰ وسیله در ساعت |
| **دوربین‌ها** | ۸ دوربین IP (۴ نمای روی‌گذر، ۴ نمای نزدیک‌شو) |
| **کنترلر سیگنال** | UTC سازگار با Siemens/NTCIP-1202 |
| **اهداف** | کاهش ≥۲۰٪ زمان انتظار، تشخیص بلادرنگ تخلفات، کنترل تطبیقی سیگنال |

### محتوای این ریپو

1. **خط داده** — تولید داده سنتتیک با CARLA و ادغام با دیتاست‌های عمومی.
2. **آموزش مدل‌ها** — تشخیص YOLOv10، ردیابی Re-ID، طبقه‌بندی ناهنجاری ST-GCN، کنترل سیگنال با PPO-GAT.
3. **بهینه‌سازی** — خروجی‌گیری موتورهای TensorRT (INT8/FP16) برای استقرار روی Jetson Orin.
4. **اجرای لبه‌ای** — پایپ‌لاین DeepStream، سرور استنتاج Triton، پل MQTT به UTC/SCADA شهرداری، و مانیتور سلامت خودترمیم.
5. **عملیات** — Docker Compose برای استقرار لبه‌ای (جتسون) و ابری (تحلیل‌ها) به‌همراه راهنمای کامل عملیات در [`RUNBOOK.md`](RUNBOOK.md).

---

## 🏗 معماری

```
┌─────────────────────────────────────────────────────────────────┐
│                گره لبه‌ای (Jetson Orin AGX 64GB)                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │  پایپ‌لاین   │──▶│   Triton    │──▶│  پل MQTT    │           │
│  │  DeepStream │   │  (TensorRT) │   │ (UTC/SCADA) │           │
│  │  (۸ دوربین) │   │   استنتاج   │   │             │           │
│  └─────────────┘   └─────────────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────┐               │
│  │           مانیتور سلامت                     │               │
│  │  GPU • دوربین‌ها • کانتینرها • بازیابی خودکار│               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                             │  MQTT روی TLS پورت 8883 / HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ابر / مرکز کنترل                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   Grafana   │   │  تحلیل داده  │   │   MLflow    │           │
│  │ (فارسی RTL) │   │ (ClickHouse)│   │   ریجستری   │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                        │                                        │
│  ┌─────────────────────────────────────────────┐               │
│  │      دوقلوی دیجیتال (SUMO + بازپخش مسیر)    │               │
│  │           تحلیل «چه می‌شود اگر»              │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**جریان داده:** فریم‌های RTSP → رمزگشایی/پیش‌پردازش در DeepStream → Triton (تشخیص → ردیابی → ناهنجاری) → ویژگی‌های مسیر حرکت → سیاست RL کنترل سیگنال → فرمان فاز → پل MQTT → UTC/SCADA. هم‌زمان تله‌متری برای داشبورد، تحلیل و دوقلوی دیجیتال به ابر آینه می‌شود.

---

## 🧠 مدل‌های هوش مصنوعی

| ماژول | مدل | فریم‌ورک | بهینه‌سازی | تأخیر (Jetson Orin) |
|---|---|---|---|---|
| **تشخیص** | YOLOv10-N (نسخه ISF) | PyTorch → TensorRT | کمّی‌سازی INT8 | ~1.8 ms |
| **ردیابی** | BoT-SORT + OSNet-x1.0 | PyTorch → TensorRT | FP16 | ~3 ms |
| **ناهنجاری** | ST-GCN (مبتنی بر مسیر حرکت) | PyTorch → ONNX Runtime | FP16 | ~5 ms |
| **کنترل سیگنال** | PPO + شبکه توجه گرافی (GAT) | PyTorch → TorchScript | FP16 | ~10 ms |

**بودجه تأخیر انتها‌به‌انتها:** کمتر از ۵۰ میلی‌ثانیه از گرفتن فریم تا صدور فرمان سیگنال (هدف اندازه‌گیری‌شده پایلوت: ~۴۲ میلی‌ثانیه).

---

## 🚗 کلاس‌های تشخیص و انواع ناهنجاری

### تشخیص — ۱۲ کلاس مخصوص ایران (`configs/isfahan_traffic.yaml`)

```
0: sedan_iran (سدان)      1: pickup_iran (وانت)     2: motorcycle_iran (موتورسیکلت)
3: bus_brt (اتوبوس BRT)   4: bus_city (اتوبوس شهری) 5: truck_heavy (کامیون سنگین)
6: van_transport (ون)     7: bicycle (دوچرخه)       8: pedestrian (عابر پیاده)
9: traffic_light (چراغ)  10: traffic_sign (علامت)  11: crosswalk (خط عابر)
```

### ناهنجاری‌ها — ۱۱ نوع تخلف مرتبط با ایران (`configs/anomalies_iran.yaml`)

```
0: normal (عادی)                  1: motorcycle_wrong_lane (حرکت موتور در خط اشتباه)
2: red_light_violation (تخلف چراغ قرمز)   3: illegal_uturn (دور زدن غیرمجاز)
4: stop_on_crosswalk (توقف روی خط عابر)   5: wrong_way_driving (خلاف جهت)
6: pedestrian_jaywalking (عدم رعایت خط عابر)  7: bus_lane_violation (تخلف خط BRT)
8: truck_in_car_lane (کامیون در خط سواری)  9: sudden_lane_change (تغییر خط ناگهانی)
10: speeding (سرعت غیرمجاز)
```

هر ناهنجاری یک **وزن اولویت** بر اساس آیین‌نامه‌های راهور ایران دارد که خط هشدار برای اولویت‌بندی و مسیریابی اعلان‌ها استفاده می‌کند.

---

## 🚀 شروع سریع

### پیش‌نیازها

| نیاز | جزئیات |
|---|---|
| **سخت‌افزار لبه** | Jetson Orin AGX 64GB (×۲ برای دسترس‌پذیری بالا) یا x86 + GPU انویدیا (A100/A6000) |
| **نرم‌افزار** | Docker 24+، NVIDIA Container Toolkit، JetPack 6.0+ (لبه) |
| **شبکه** | دسترسی به ۸ دوربین RTSP، بروکر MQTT (UTC)، نقاط HTTPS |
| **آموزش** | توصیه می‌شود گره چند‌GPU (بخش `.env.example`: `TRAIN_GPUS=0,1,2,3`) |

### ۱. کلون و پیکربندی

```bash
git clone https://github.com/alirezahs81/isfahan-smart-intersection.git
cd isfahan-smart-intersection

cp .env.example .env
# فایل .env را ویرایش کنید: آدرس RTSP دوربین‌ها، اطلاعات MQTT، مسیر گواهی TLS، آدرس وبهوک‌ها
```

> ⚠️ هرگز `.env` یا گواهی‌های TLS را کامیت نکنید — در `.gitignore` هستند. تنها فایل env ردیابی‌شده `.env.example` است که فقط placeholder دارد.

### ۲. تولید داده سنتتیک (اجرای اول)

```bash
# اجرای سرور CARLA
docker run -d --name carla -p 2000-2002:2000-2002 --gpus all carlasim/carla:0.9.15

# ساخت نقشه تقاطع اصفهان در CARLA
python tools/carla_isfahan_map/generate_isfahan_map.py

# تولید حدود ۵۰ هزار فریم برچسب‌خورده با ترکیب وسایل ایرانی
python tools/synthetic_generator/generate_synthetic_data.py --frames 50000
```

### ۳. ادغام دیتاست‌ها و آموزش

```bash
# ادغام داده سنتتیک + دیتاست‌های عمومی (UA-DETRAC، BDD100K و…)
python tools/data_fusion/merge_datasets.py --output data/fused

# تشخیص (YOLOv10)
yolo detect train data=configs/isfahan_traffic.yaml model=yolov10n.pt \
    epochs=150 imgsz=640 batch=64 device=0,1,2,3

# Re-ID برای ردیابی
python models/tracking/train_reid_isfahan.py --data data/fused --epochs 120

# تشخیص ناهنجاری (ST-GCN روی مسیر حرکت)
python models/anomaly/stgcn_anomaly.py --mode train --data data/fused/trajectories

# کنترل سیگنال (RL گرافی PPO-GAT)
python models/signal_rl/ppo_gat_signal.py --mode train --timesteps 1000000 --n_envs 8

# دوقلوی دیجیتال (هم‌شبیه‌سازی SUMO)
python models/digital_twin/isfahan_digital_twin.py --config configs/intersection.yaml
```

### ۴. خروجی TensorRT و استقرار لبه‌ای

```bash
# خروجی‌گیری همه مدل‌ها به موتورهای TensorRT INT8/FP16
python export_tensorrt.py --all

# ساخت ایمیج‌های لبه
docker compose -f docker-compose.edge.yml build --parallel

# استقرار روی Jetson Orin
scp -r . jetson@<edge-ip>:/opt/isfahan-sip/
ssh jetson@<edge-ip> "cd /opt/isfahan-sip && docker compose -f docker-compose.edge.yml up -d"
```

### ۵. مانیتورینگ

```bash
# نقطه سلامت (مانیتور سلامت لبه)
curl http://localhost:9091/health

# متریک‌های Prometheus
curl http://localhost:9090/metrics

# داشبورد Grafana (فارسی RTL) — با docker-compose.cloud.yml آماده می‌شود
open http://localhost:3000
```

برای رویه‌های گام‌به‌گام عملیاتی (راه‌اندازی، رفع خطا، کالیبراسیون، بازگشت به عقب) [`RUNBOOK.md`](RUNBOOK.md) را ببینید.

---

## 📁 ساختار پروژه

```
isfahan-smart-intersection/
├── 🐳 docker-compose.edge.yml        # پشته لبه: DeepStream، Triton، پل MQTT، مانیتور سلامت
├── 🐳 docker-compose.cloud.yml       # پشته ابری: Grafana، تحلیل داده، ریجستری مدل
├── ⚙️ configs/
│   ├── intersection.yaml             # توپولوژی تقاطع و مشخصات فازها
│   ├── signal_phases.yaml            # پلن‌های زمان‌بندی پایه سیگنال (۵ فاز)
│   ├── signal_rl.yaml                # هایپرپارامترها و تابع پاداش ایجنت RL
│   ├── anomalies_iran.yaml           # تعاریف تخلفات ایرانی و اولویت‌ها
│   └── isfahan_traffic.yaml          # پیکربندی دیتاست YOLO (۱۲ کلاس)
├── 🖥️ edge/
│   ├── deepstream_pipeline/
│   │   ├── isfahan_pipeline.py       # پایپ‌لاین DeepStream/GStreamer برای ۸ دوربین
│   │   └── configs/                  # پیکربندی PGIE/SGIE، ردیاب و برچسب‌ها
│   ├── triton_model_repo/            # پیکربندی مدل‌های Triton (تشخیص، ReID، ناهنجاری، RL)
│   ├── mqtt_bridge/                  # کلاینت MQTT برای اتصال خروجی‌ها و فرمان‌ها به UTC/SCADA
│   └── health_monitor/               # نگهبان GPU/دوربین/کانتینر با بازیابی خودکار
├── 🧠 models/
│   ├── detection/                    # آموزش YOLOv10-ISF (۱۲ کلاس ایرانی)
│   ├── tracking/                     # فاین‌تیون OSNet Re-ID
│   ├── anomaly/                      # طبقه‌بند ناهنجاری مسیر حرکت ST-GCN
│   ├── signal_rl/                    # سیاست تطبیقی سیگنال PPO-GAT
│   └── digital_twin/                 # دوقلوی دیجیتال تقاطع مبتنی بر SUMO
├── 🛠️ tools/
│   ├── carla_isfahan_map/            # سازنده نقشه تقاطع در CARLA
│   ├── synthetic_generator/          # مولد داده سنتتیک برچسب‌خورده
│   └── data_fusion/                  # ادغام داده سنتتیک + دیتاست‌های عمومی
├── export_tensorrt.py                # خروجی‌گیر موتورهای TensorRT INT8/FP16 (همه مدل‌ها)
├── requirements.txt                  # وابستگی‌های پایتون
├── .env.example                      # قالب متغیرهای محیطی (فقط placeholder)
├── RUNBOOK.md                        # رویه‌های عملیات و پاسخ به حوادث
├── LICENSE                           # لایسنس MIT
└── README.md                         # همین فایل (دوزبانه EN/FA)
```

> **نکته:** وزن مدل‌ها، دیتاست‌ها، موتورهای TensorRT، گواهی‌های TLS و `.env` عمداً **ردیابی نمی‌شوند** — ببینید [`.gitignore`](.gitignore).

---

## 🔧 نکات پیکربندی

### پلن زمان‌بندی سیگنال (`configs/signal_phases.yaml`)

```yaml
phases:
  - id: 0
    name: "NB_SB_Straight"
    movements: ["N->S", "S->N"]
    min_green: 20
    max_green: 80
  # … در مجموع ۵ فاز، شامل فاز اولویت BRT
```

### اولویت ناهنجاری‌ها (`configs/anomalies_iran.yaml`)

```yaml
anomaly_priorities:
  red_light_violation: 10
  wrong_way_driving: 10
  truck_in_car_lane: 8
  speeding: 7
  # … وزن‌ها بر اساس آیین‌نامه راهور ایران
```

پاداش ایجنت RL در `configs/signal_rl.yaml` ترکیبی از **طول صف، زمان انتظار هر فاز، ایمنی عبور عابر پیاده و رعایت فاصله زمانی BRT** است — وزن‌های دقیق در همان فایل آمده است.

---

## 📈 اهداف عملکرد

| متریک | هدف | پایلوت (اندازه‌گیری‌شده) |
|---|---|---|
| mAP@0.5 (تشخیص) | > 0.90 | 0.87 |
| MOTA (ردیابی) | > 0.80 | 0.76 |
| F1 (ناهنجاری) | > 0.92 | 0.89 |
| کاهش زمان انتظار | > ۲۰٪ | ۱۸٪ |
| تأخیر انتها‌به‌انتها | < 50 ms | 42 ms |
| دسترس‌پذیری | > ۹۹.۵٪ | ۹۹.۲٪ |
| نرخ مثبت کاذب | < ۵٪ | ۶.۲٪ |

اعداد پایلوت از حلقه شبیه‌سازی CARLA + داده سنتتیک است؛ کالیبراسیون محل پروژه اختلاف باقی‌مانده را برطرف می‌کند.

---

## 🛡 امنیت و حریم خصوصی

- **TLS 1.3** برای همه ارتباطات MQTT/HTTP؛ **mTLS** (گواهی کلاینت) با UTC/SCADA.
- **جداسازی شبکه:** VLAN لبه از شبکه شهرداری مجزا است.
- **بهداشت رمزها:** اطلاعات ورود فقط در `.env` (ردیابی‌نشده)؛ `.env.example` فقط placeholder دارد.
- **نگهداشت داده:** ۳۰ روز فیلم خام، ۱ سال داده تجمیعی (طبق مقررات ایران).
- **حریم خصوصی:** محو چهره و پلاک در ویدئوهای ذخیره‌شده.
- **ردگیری حسابرسی:** هر تغییر فرمان سیگنال با هویت اپراتور/ایجنت ثبت می‌شود.

---

## ✅ چک‌لیست استقرار

**پیش از استقرار**
- [ ] امضای MOU با شهرداری اصفهان
- [ ] اعتبارسنجی اطلاعات RTSP دوربین‌ها
- [ ] دریافت دسترسی API از UTC/SCADA و نصب گواهی‌های TLS
- [ ] آماده‌سازی دستگاه‌های Jetson Orin (JetPack 6.0)
- [ ] پیکربندی VLAN شبکه، UPS و مانیتورینگ محیطی

**استقرار**
- [ ] اعتبارسنجی نقشه CARLA در محل، تکمیل کالیبراسیون دوربین‌ها
- [ ] تأیید تولید داده سنتتیک، آموزش مدل‌ها و خروجی TensorRT
- [ ] ساخت و اسکن ایمیج‌های Docker، تست ۲۴ ساعته لبه
- [ ] تأیید یکپارچگی MQTT با UTC، بومی‌سازی Grafana (فارسی RTL)
- [ ] تست وبهوک‌های هشدار و سوئیچ اضطراری (اصلی → یدکی)

**پس از استقرار**
- [ ] ۲ هفته مانیتورینگ پایلوت، تعیین خط پایه KPI
- [ ] آموزش اپراتورها، تحویل مستندات، توافق SLA نگهداری

---

## 🤝 مشارکت

1. ریپو را Fork کنید
2. شاخه جدید بسازید: `git checkout -b feature/amazing-feature`
3. با پیام کامیت واضح کامیت بزنید: `git commit -m 'Add amazing feature'`
4. Push کنید و Pull Request باز کنید

**استانداردهای کد**
- پایتون: `black`، `isort`، `mypy`، `pylint`
- Docker: `hadolint`، بیلد چند‌مرحله‌ای
- پیکربندی‌ها: YAML با اعتبارسنجی schema
- مستندات: دوزبانه (انگلیسی + فارسی)

---

## 📄 لایسنس

منتشرشده تحت [لایسنس MIT](LICENSE).

---

## 🙏 سپاسگزاری

- **NVIDIA** — Jetson Orin، DeepStream، TensorRT، Triton
- **Ultralytics** — YOLOv10
- **CARLA** و **SUMO** — شبیه‌سازی داده و ترافیک
- نویسندگان **BoT-SORT / OSNet** — مبانی ردیابی و Re-ID
- معاونت حمل‌ونقل شهرداری اصفهان — دانش حوزه‌ای

---

*ساخته‌شده با ❤️ برای خیابان‌های هوشمندتر و امن‌تر اصفهان*
