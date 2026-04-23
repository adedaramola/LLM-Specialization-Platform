"""Tests for emit_metrics_json — CI gate mode separation."""
import json
import pytest

from src.evaluation.harness import emit_metrics_json


def _result(label, raw, constrained=None):
    return {
        "model": label,
        "model_path": f"./artifacts/{label}",
        "provider": "hf_native",
        "raw": raw,
        "constrained": constrained or {},
        "raw_vs_guided_gap": {},
        "n_examples": 100,
        "n_positive": 86,
        "n_null": 14,
    }


RAW_THRESHOLDS = {"schema_validity": 0.90, "field_f1": 0.80, "null_accuracy": 0.85}
GATE_THRESHOLDS = {"schema_validity": 0.90, "field_f1": 0.50, "null_accuracy": 0.90}

GATE_CFG_CONSTRAINED = {
    "mode": "constrained",
    "fallback_to_raw": True,
    "thresholds": GATE_THRESHOLDS,
}
GATE_CFG_RAW = {
    "mode": "raw",
    "fallback_to_raw": True,
    "thresholds": GATE_THRESHOLDS,
}


class TestEmitMetricsJsonGateMode:
    def test_ci_pass_uses_constrained_when_available(self, tmp_path):
        # Raw fails schema_validity (0.40); constrained passes (0.92).
        # Gate = constrained → ci_pass should be True.
        results = [_result("dpo",
            raw={"schema_validity": 0.40, "field_f1": 0.79, "null_accuracy": 0.95},
            constrained={"schema_validity": 0.92, "field_f1": 0.55, "null_accuracy": 0.95},
        )]
        out = emit_metrics_json(
            results, RAW_THRESHOLDS, str(tmp_path / "m.json"),
            gate_cfg=GATE_CFG_CONSTRAINED,
        )
        assert out["ci_pass"] is True
        assert out["ci_gate_mode"] == "constrained"

    def test_ci_fails_when_constrained_below_gate_threshold(self, tmp_path):
        # Constrained field_f1 = 0.30 < gate threshold 0.50 → ci_pass False
        results = [_result("dpo",
            raw={"schema_validity": 0.95, "field_f1": 0.85, "null_accuracy": 0.95},
            constrained={"schema_validity": 0.95, "field_f1": 0.30, "null_accuracy": 0.95},
        )]
        out = emit_metrics_json(
            results, RAW_THRESHOLDS, str(tmp_path / "m.json"),
            gate_cfg=GATE_CFG_CONSTRAINED,
        )
        assert out["ci_pass"] is False

    def test_raw_pass_fail_independent_of_gate(self, tmp_path):
        # Raw fails schema_validity but constrained passes it.
        # pass_fail must reflect raw; deployment_pass_fail must reflect constrained.
        results = [_result("dpo",
            raw={"schema_validity": 0.40, "field_f1": 0.60, "null_accuracy": 0.95},
            constrained={"schema_validity": 0.93, "field_f1": 0.55, "null_accuracy": 0.95},
        )]
        out = emit_metrics_json(
            results, RAW_THRESHOLDS, str(tmp_path / "m.json"),
            gate_cfg=GATE_CFG_CONSTRAINED,
        )
        model = out["models"]["dpo"]
        # Raw pass_fail: schema_validity 0.40 < 0.90 → failed
        assert model["pass_fail"]["schema_validity"]["passed"] is False
        # Deployment pass_fail: constrained schema_validity 0.93 >= 0.90 → passed
        assert model["deployment_pass_fail"]["schema_validity"]["passed"] is True
        assert model["deployment_gate_mode"] == "constrained"

    def test_fallback_to_raw_when_constrained_empty(self, tmp_path):
        # No constrained metrics; gate should fall back to raw.
        results = [_result("sft",
            raw={"schema_validity": 0.95, "field_f1": 0.85, "null_accuracy": 0.92},
        )]
        out = emit_metrics_json(
            results, RAW_THRESHOLDS, str(tmp_path / "m.json"),
            gate_cfg=GATE_CFG_CONSTRAINED,
        )
        assert out["ci_pass"] is True
        assert out["models"]["sft"]["deployment_gate_mode"] == "raw"

    def test_gate_mode_raw_uses_raw_regardless(self, tmp_path):
        # Gate=raw; even if constrained is present, raw drives ci_pass.
        # Raw null_accuracy 0.70 < gate threshold 0.90 → ci_pass False
        results = [_result("dpo",
            raw={"schema_validity": 0.95, "field_f1": 0.85, "null_accuracy": 0.70},
            constrained={"schema_validity": 0.98, "field_f1": 0.60, "null_accuracy": 0.98},
        )]
        out = emit_metrics_json(
            results, RAW_THRESHOLDS, str(tmp_path / "m.json"),
            gate_cfg=GATE_CFG_RAW,
        )
        assert out["ci_pass"] is False
        assert out["models"]["dpo"]["deployment_gate_mode"] == "raw"

    def test_output_written_to_disk(self, tmp_path):
        results = [_result("sft",
            raw={"schema_validity": 0.95, "field_f1": 0.85, "null_accuracy": 0.92},
        )]
        path = str(tmp_path / "sub" / "metrics.json")
        emit_metrics_json(results, RAW_THRESHOLDS, path, gate_cfg=GATE_CFG_CONSTRAINED)
        with open(path) as f:
            on_disk = json.load(f)
        assert "models" in on_disk
        assert "ci_gate_mode" in on_disk

    def test_no_gate_cfg_defaults_to_raw(self, tmp_path):
        # Backward-compatible: no gate_cfg → raw mode, ci_pass based on raw
        results = [_result("sft",
            raw={"schema_validity": 0.40, "field_f1": 0.85, "null_accuracy": 0.92},
        )]
        out = emit_metrics_json(
            results, RAW_THRESHOLDS, str(tmp_path / "m.json"),
        )
        # Raw schema_validity 0.40 < 0.90 → ci_pass False (raw gate)
        assert out["ci_pass"] is False
