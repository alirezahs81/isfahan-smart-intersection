# Repository status — Isfahan Smart Intersection (ISIP)

**Last updated:** 2026-11-23

[![CI](https://github.com/alirezahs81/isfahan-smart-intersection/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/isfahan-smart-intersection/actions/workflows/ci.yml)

## Software / GitHub readiness

| Area | Status |
|------|--------|
| Reference implementation (models, edge, configs, ops) | Complete |
| Bilingual README + RUNBOOK | Complete |
| `.env.example` + secrets hygiene | Complete |
| Docker Compose (edge + cloud) | Complete |
| Unit tests (`tests/`) | Complete |
| GitHub Actions CI | Passing |
| CONTRIBUTING + Makefile | Complete |
| Open issues / PRs | None |

Local verification:

```bash
pip install -r requirements-ci.txt
make ci
```

## Field pilot (outside this repo)

These require municipal access, hardware, and on-site work — not tracked in Git:

- Model weights / TensorRT engines (see `.gitignore`)
- Live RTSP cameras and calibration files
- Real `.env`, TLS certificates, UTC/SCADA credentials
- Jetson Orin provisioning and 24h soak test
- MOU, operator training, KPI baseline (see Deployment Checklist in README)

**Conclusion:** The repository is complete as a reference implementation. Production go-live depends on the field checklist above.
