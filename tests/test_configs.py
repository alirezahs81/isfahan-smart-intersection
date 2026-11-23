"""Config integrity tests for Isfahan Smart Intersection pilot."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load_yaml(name: str) -> dict:
    path = CONFIGS / name
    assert path.exists(), f"Missing config: {path}"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, f"Empty config: {path}"
    return data


def test_intersection_topology():
    data = load_yaml("intersection.yaml")
    inter = data["intersection"]

    assert inter["id"] == "chaharbag_azadi"
    assert inter["city"] == "Isfahan"
    lat, lon = inter["coordinates"]
    assert 32.0 < lat < 33.0
    assert 51.0 < lon < 52.0

    geometry = inter["geometry"]
    assert geometry["lanes_per_approach"] == 4
    assert geometry["has_brt_lane"] is True
    assert geometry["pedestrian_crossings"] == 4

    controller = inter["signal_controller"]
    assert controller["phases"] == 5
    assert controller["min_green"] < controller["max_green"]
    assert controller["yellow_time"] > 0
    assert controller["all_red_time"] > 0

    cameras = inter["cameras"]
    assert len(cameras) == 8
    ids = {c["id"] for c in cameras}
    assert ids == {f"cam_{i:02d}" for i in range(1, 9)}
    for cam in cameras:
        assert cam["fps"] == 25
        assert cam["resolution"] == [1920, 1080]
        assert "rtsp://" in cam["rtsp"]


def test_signal_phases():
    data = load_yaml("signal_phases.yaml")
    phases = data["phases"]

    assert len(phases) == 5
    assert {p["id"] for p in phases} == {0, 1, 2, 3, 4}

    for phase in phases:
        assert phase["min_green"] <= phase["max_green"]
        assert phase["min_green"] > 0
        assert phase["all_red"] >= 0
        assert "name" in phase and "name_fa" in phase
        assert isinstance(phase["movements"], list)
        assert len(phase["movements"]) >= 1

    pedestrian = next(p for p in phases if p["id"] == 4)
    assert pedestrian["pedestrian_crossing"] is True
    assert pedestrian["priority"] == "pedestrian"

    adaptive = data["adaptive_config"]
    assert adaptive["cycle_time_min"] < adaptive["cycle_time_max"]
    assert adaptive["min_green_enforced"] is True

    priorities = data["direction_priorities"]
    for key in ("north", "south", "east", "west", "pedestrian"):
        assert key in priorities
        assert priorities[key] > 0


def test_isfahan_traffic_yolo_config():
    data = load_yaml("isfahan_traffic.yaml")

    assert data["nc"] == 12
    names = data["names"]
    assert len(names) == 12
    assert set(names.keys()) == set(range(12))

    expected = {
        0: "sedan_iran",
        1: "pickup_iran",
        2: "motorcycle_iran",
        3: "bus_brt",
        4: "bus_city",
        5: "truck_heavy",
        6: "van_transport",
        7: "bicycle",
        8: "pedestrian",
        9: "traffic_light",
        10: "traffic_sign",
        11: "crosswalk",
    }
    assert names == expected

    weights = data["class_weights"]
    for name in expected.values():
        assert name in weights
        assert weights[name] > 0

    train_cfg = data["train_cfg"]
    assert train_cfg["epochs"] > 0
    assert train_cfg["imgsz"] == 640
    assert train_cfg["batch"] > 0


def test_anomalies_iran():
    data = load_yaml("anomalies_iran.yaml")
    classes = data["anomaly_classes"]

    assert len(classes) == 11
    ids = [c["id"] for c in classes]
    assert ids == list(range(11))

    names = [c["name"] for c in classes]
    assert names[0] == "normal"
    assert "red_light_violation" in names
    assert "wrong_way_driving" in names

    for c in classes:
        assert "name_fa" in c
        assert "severity" in c
        assert c["severity"] >= 0
        assert "enforcement_action" in c
        assert "fine_toman" in c

    # Critical severities
    by_name = {c["name"]: c for c in classes}
    assert by_name["red_light_violation"]["severity"] == 10
    assert by_name["wrong_way_driving"]["severity"] == 10

    priorities = data["alert_priorities"]
    assert 2 in priorities["critical"]  # red_light
    assert 5 in priorities["critical"]  # wrong_way

    actions = data["automated_actions"]
    for level in ("critical", "high", "medium", "low"):
        assert level in actions
        assert isinstance(actions[level], list)
        assert len(actions[level]) >= 1


def test_signal_rl_config_exists():
    data = load_yaml("signal_rl.yaml")
    assert isinstance(data, dict)
    assert len(data) > 0


def test_all_config_files_parse():
    """Every YAML under configs/ must be valid and non-empty."""
    paths = sorted(CONFIGS.rglob("*.yaml")) + sorted(CONFIGS.rglob("*.yml"))
    assert paths, "No config files found"
    for path in paths:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None, f"Empty or invalid YAML: {path}"
