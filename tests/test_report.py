"""Tests for report generation and qualitative sample categorisation."""
import json
import pytest
from pathlib import Path

from src.evaluation.report import generate_report, _metric, _delta, _fmt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

METRICS_DATA = {
    "schema_version": "1.0.0",
    "ci_pass": True,
    "models": {
        "base": {"raw": {"field_f1": 0.42, "null_accuracy": 0.60, "schema_validity": 0.70}, "constrained": {}, "pass_fail": {}},
        "sft":  {"raw": {"field_f1": 0.78, "null_accuracy": 0.82, "schema_validity": 0.91}, "constrained": {"field_f1": 0.85}, "pass_fail": {}},
        "dpo":  {"raw": {"field_f1": 0.83, "null_accuracy": 0.88, "schema_validity": 0.94}, "constrained": {}, "pass_fail": {}},
        "merged_bf16": {"raw": {"field_f1": 0.82, "null_accuracy": 0.87, "schema_validity": 0.93}, "constrained": {}, "pass_fail": {}},
        "gguf_q8":  {"raw": {"field_f1": 0.81, "null_accuracy": 0.86, "schema_validity": 0.93}, "constrained": {}, "pass_fail": {}},
        "gguf_q4k": {"raw": {"field_f1": 0.77, "null_accuracy": 0.84, "schema_validity": 0.91}, "constrained": {}, "pass_fail": {}},
    },
}

CONFIG = {
    "_config_path": "configs/sft_config.yaml",
    "pipeline_version": "1.0.0",
    "task": {"name": "json_extraction", "description": "Extract structured JSON"},
    "model": {"name": "Qwen/Qwen2.5-7B-Instruct", "license": "apache-2.0",
               "license_restrictions": "Attribution required", "attribution": "Alibaba Cloud"},
    "dataset": {"sources": ["synthetic"], "licenses": ["cc-by-4.0"],
                 "null_case_fraction": 0.15},
    "lora": {"rank": 32, "alpha": 32, "dropout": 0.05, "use_rslora": False,
              "target_modules": ["q_proj", "v_proj"]},
    "training": {"learning_rate": 1e-4, "num_train_epochs": 2,
                  "lr_scheduler_type": "cosine", "bf16": True,
                  "per_device_train_batch_size": 4, "gradient_accumulation_steps": 8,
                  "gradient_checkpointing": True, "output_dir": "./artifacts/dpo"},
    "dpo": {"beta": 0.1, "precompute_ref_log_probs": True},
    "preference_data": {"target_pairs": 2000, "null_case_fraction": 0.20,
                         "ranking_strategy": "deterministic"},
    "reproducibility": {"seed": 42, "tolerances": {"f1": 0.015, "null_accuracy": 0.020}},
}


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestFmt:
    def test_float(self):
        assert _fmt(0.832) == "0.832"

    def test_none(self):
        assert _fmt(None) == "—"

    def test_int(self):
        assert _fmt(42) == "42"


class TestMetricLookup:
    def test_raw_metric(self):
        assert _metric(METRICS_DATA, "sft", "field_f1") == "0.780"

    def test_constrained_metric(self):
        assert _metric(METRICS_DATA, "sft", "field_f1", "constrained") == "0.850"

    def test_missing_model(self):
        assert _metric(METRICS_DATA, "nonexistent", "field_f1") == "—"


class TestDelta:
    def test_positive_delta(self):
        val = _delta(METRICS_DATA, "base", "sft", "field_f1")
        assert val == "0.360"

    def test_negative_delta(self):
        val = _delta(METRICS_DATA, "dpo", "merged_bf16", "field_f1")
        assert val == "-0.010"

    def test_missing_model(self):
        assert _delta(METRICS_DATA, "base", "ghost", "field_f1") == "—"


# ---------------------------------------------------------------------------
# Integration test: report generation
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_renders_template(self, tmp_path):
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(METRICS_DATA))

        output_path = tmp_path / "report.md"
        template_path = Path("templates/model_card.md")
        if not template_path.exists():
            pytest.skip("templates/model_card.md not found — run from project root")

        rendered = generate_report(
            template_path=str(template_path),
            metrics_json_path=str(metrics_path),
            manifest_path=None,
            config=CONFIG,
            output_path=str(output_path),
        )

        assert output_path.exists()
        assert "Qwen2.5-7B-Instruct" in rendered
        assert "apache-2.0" in rendered
        assert "0.780" in rendered   # sft field_f1
        assert "0.830" in rendered   # dpo field_f1

    def test_no_unfilled_critical_tokens(self, tmp_path):
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(METRICS_DATA))
        output_path = tmp_path / "report.md"
        template_path = Path("templates/model_card.md")
        if not template_path.exists():
            pytest.skip("templates/model_card.md not found — run from project root")

        rendered = generate_report(
            template_path=str(template_path),
            metrics_json_path=str(metrics_path),
            manifest_path=None,
            config=CONFIG,
            output_path=str(output_path),
        )
        # These must be filled — missing them is a delivery failure per CLAUDE.md
        for token in ("base_model_license", "attribution", "dataset_hash"):
            assert "{{" + token + "}}" not in rendered, f"Unfilled token: {token}"
