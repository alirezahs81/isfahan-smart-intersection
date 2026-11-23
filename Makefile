.PHONY: help install-ci test lint-syntax validate-configs ci

help:
	@echo "Isfahan Smart Intersection — common targets"
	@echo "  make install-ci      Install lightweight CI dependencies"
	@echo "  make validate-configs  Parse all YAML under configs/"
	@echo "  make test            Run unit tests"
	@echo "  make lint-syntax     compileall on Python packages"
	@echo "  make ci              Full local CI mirror"

install-ci:
	pip install -r requirements-ci.txt

test:
	pytest tests/ -v --tb=short

lint-syntax:
	python -m compileall -q models tools edge/health_monitor edge/mqtt_bridge export_tensorrt.py tests

validate-configs:
	python -c "from pathlib import Path; import yaml; \
files=list(Path('configs').rglob('*.yaml'))+list(Path('configs').rglob('*.yml')); \
assert files; \
[yaml.safe_load(p.read_text(encoding='utf-8')) for p in files]; \
print(f'OK {len(files)} config file(s)')"

ci: install-ci lint-syntax validate-configs test
