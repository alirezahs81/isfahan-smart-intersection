# Contributing to Isfahan Smart Intersection (ISIP)

Thanks for contributing. This repository is a **reference implementation** for the Chaharbagh–Azadi pilot. Keep changes practical, tested, and bilingual where documentation is involved.

## Development setup

```bash
git clone https://github.com/alirezahs81/isfahan-smart-intersection.git
cd isfahan-smart-intersection
cp .env.example .env   # never commit real secrets

# Lightweight CI toolchain (no GPU / DeepStream required)
pip install -r requirements-ci.txt
make ci
```

Full training and edge stacks use `requirements.txt` and the per-service files under `edge/*/requirements.txt`. Those need NVIDIA GPUs / JetPack and are **not** required for config or unit-test work.

## Branch & PR workflow

1. Create a feature branch: `git checkout -b feature/short-description`
2. Keep commits focused and descriptive
3. Run `make ci` before opening a PR
4. Open a PR against `main` with a short summary of *what* and *why*

## Code standards

| Area | Standard |
|------|----------|
| Python formatting | `black`, `isort` |
| Types | `mypy` preferred on new pure-Python modules |
| Configs | Valid YAML; update `tests/` when schemas change |
| Docs | English + Persian for user-facing README sections |
| Secrets | Only placeholders in `.env.example`; never commit `.env` or certs |

## Testing policy

- **Required for CI:** config integrity and pure-logic unit tests under `tests/`
- **Edge modules** (`mqtt_bridge`, `health_monitor`, DeepStream, Triton) depend on hardware/services — mock or pure-logic tests only in GitHub Actions
- When you change anomaly classes, signal phases, or YOLO class lists, update the matching tests in the same PR

## What not to commit

- Model weights (`.pt`, `.engine`, `.onnx`, …)
- Datasets, video, TensorRT caches
- TLS certificates and real credentials
- `docker-compose.override.yml` with local secrets

See [`.gitignore`](.gitignore) for the full list.

## License

By contributing, you agree that your contributions are licensed under the MIT License (see [LICENSE](LICENSE)).
