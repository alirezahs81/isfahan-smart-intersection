"""Cross-check anomaly definitions against the ST-GCN model constants."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ANOMALY_YAML = ROOT / "configs" / "anomalies_iran.yaml"
STGCN_PATH = ROOT / "models" / "anomaly" / "stgcn_anomaly.py"
INTERSECTION_YAML = ROOT / "configs" / "intersection.yaml"


def _load_anomaly_yaml() -> dict:
    with ANOMALY_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_stgcn_constants() -> tuple[list[str], dict[int, int]]:
    """Parse ANOMALY_CLASSES and SEVERITY_WEIGHTS from source without importing torch."""
    source = STGCN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    classes: list[str] | None = None
    severity: dict[int, int] | None = None

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "IsfahanSTGCN":
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign):
                continue
            for target in item.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "ANOMALY_CLASSES":
                    classes = ast.literal_eval(item.value)
                elif target.id == "SEVERITY_WEIGHTS":
                    severity = ast.literal_eval(item.value)

    assert classes is not None, "ANOMALY_CLASSES not found in IsfahanSTGCN"
    assert severity is not None, "SEVERITY_WEIGHTS not found in IsfahanSTGCN"
    return classes, severity


def test_stgcn_classes_match_yaml():
    data = _load_anomaly_yaml()
    yaml_names = [c["name"] for c in data["anomaly_classes"]]
    model_classes, _ = _extract_stgcn_constants()

    assert len(yaml_names) == 11
    assert yaml_names == model_classes


def test_stgcn_severity_matches_yaml():
    data = _load_anomaly_yaml()
    _, model_severity = _extract_stgcn_constants()

    for c in data["anomaly_classes"]:
        assert model_severity[c["id"]] == c["severity"], (
            f"Severity mismatch for id={c['id']} name={c['name']}: "
            f"model={model_severity[c['id']]} yaml={c['severity']}"
        )


def test_intersection_priority_keys_are_known_anomalies():
    inter = yaml.safe_load(INTERSECTION_YAML.read_text(encoding="utf-8"))
    priorities = inter["intersection"]["anomaly_priorities"]
    data = _load_anomaly_yaml()
    known = {c["name"] for c in data["anomaly_classes"] if c["name"] != "normal"}

    for name in priorities:
        assert name in known, f"Unknown anomaly priority key: {name}"

    # High-severity violations must be present
    for critical in ("red_light_violation", "wrong_way_driving"):
        assert critical in priorities
        assert priorities[critical] >= 8
