"""Pure-logic tests for edge MQTT transforms and health status rules.

These tests intentionally do NOT import paho-mqtt / docker / NVML so CI stays
lightweight. Behaviour mirrors edge/mqtt_bridge and edge/health_monitor.
"""

from __future__ import annotations

import time
from typing import Any


# --- Mirrors IsfahanMQTTBridge._add_severity_priority ---
SEVERITY_MAP = {
    "red_light_violation": "critical",
    "wrong_way_driving": "critical",
    "truck_in_car_lane": "high",
    "speeding": "high",
    "sudden_lane_change": "medium",
    "bus_lane_violation": "medium",
    "illegal_uturn": "medium",
    "stop_on_crosswalk": "low",
    "motorcycle_wrong_lane": "low",
    "pedestrian_jaywalking": "low",
}

SCORE_MAP = {"critical": 10, "high": 7, "medium": 5, "low": 3, "info": 1}


def add_severity_priority(payload: dict) -> dict:
    anomaly_type = payload.get("anomaly", {}).get("anomaly_type", "unknown")
    payload = dict(payload)
    payload["priority"] = SEVERITY_MAP.get(anomaly_type, "info")
    payload["severity_score"] = SCORE_MAP.get(payload["priority"], 1)
    return payload


def format_cms_violation(payload: dict) -> dict:
    anomaly = payload.get("anomaly", {})
    return {
        "violation_id": payload.get("alert_id", f"viol_{int(time.time())}"),
        "violation_type": anomaly.get("anomaly_type", "unknown"),
        "location": {
            "intersection": payload.get("intersection_id", "chaharbag_azadi"),
            "camera": anomaly.get("camera_id", "unknown"),
            "zone": anomaly.get("zone", "unknown"),
        },
        "timestamp": anomaly.get("timestamp", time.time()),
        "vehicle_info": {
            "track_id": anomaly.get("track_id"),
            "class": anomaly.get("class_name", "unknown"),
        },
        "evidence": {
            "confidence": anomaly.get("confidence", 0),
            "severity": anomaly.get("severity", 0),
        },
        "status": "pending_review",
    }


def parse_signal_command(payload: dict) -> dict:
    return {
        "command": "set_signal_plan",
        "intersection_id": payload.get("intersection_id"),
        "plan_id": payload.get("plan_id"),
        "phase_durations": payload.get("phase_durations", {}),
        "priority": payload.get("priority", "normal"),
        "source": "utc",
        "timestamp": payload.get("timestamp", time.time()),
    }


def overall_health_status(component_statuses: list[str]) -> str:
    """Mirrors IsfahanHealthMonitor._health_endpoint aggregation."""
    overall = "healthy"
    for status in component_statuses:
        if status == "critical":
            return "critical"
        if status == "degraded" and overall == "healthy":
            overall = "degraded"
    return overall


def test_critical_anomalies_get_highest_priority():
    for name in ("red_light_violation", "wrong_way_driving"):
        out = add_severity_priority({"anomaly": {"anomaly_type": name}})
        assert out["priority"] == "critical"
        assert out["severity_score"] == 10


def test_unknown_anomaly_defaults_to_info():
    out = add_severity_priority({"anomaly": {"anomaly_type": "not_a_real_type"}})
    assert out["priority"] == "info"
    assert out["severity_score"] == 1


def test_cms_violation_shape():
    payload = {
        "alert_id": "a-1",
        "intersection_id": "chaharbag_azadi",
        "anomaly": {
            "anomaly_type": "speeding",
            "camera_id": "cam_01",
            "zone": "approach",
            "track_id": 42,
            "class_name": "sedan_iran",
            "confidence": 0.91,
            "severity": 7,
            "timestamp": 1_700_000_000.0,
        },
    }
    out = format_cms_violation(payload)
    assert out["violation_id"] == "a-1"
    assert out["violation_type"] == "speeding"
    assert out["location"]["camera"] == "cam_01"
    assert out["vehicle_info"]["track_id"] == 42
    assert out["status"] == "pending_review"
    assert out["evidence"]["confidence"] == 0.91


def test_signal_command_defaults():
    out = parse_signal_command({"intersection_id": "chaharbag_azadi", "plan_id": "P1"})
    assert out["command"] == "set_signal_plan"
    assert out["phase_durations"] == {}
    assert out["priority"] == "normal"
    assert out["source"] == "utc"


def test_health_aggregation_critical_wins():
    assert overall_health_status(["healthy", "degraded", "critical"]) == "critical"


def test_health_aggregation_degraded_when_no_critical():
    assert overall_health_status(["healthy", "degraded", "healthy"]) == "degraded"


def test_health_aggregation_all_healthy():
    assert overall_health_status(["healthy", "healthy"]) == "healthy"


def test_all_severity_map_keys_are_documented_anomalies():
    """Every MQTT severity key (except unknown) must exist in anomalies_iran.yaml."""
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs" / "anomalies_iran.yaml").read_text(encoding="utf-8"))
    known = {c["name"] for c in data["anomaly_classes"]}
    for name in SEVERITY_MAP:
        assert name in known, f"MQTT severity key missing from YAML: {name}"
