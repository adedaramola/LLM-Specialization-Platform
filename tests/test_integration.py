"""
Integration tests: harness orchestration end-to-end with a fake provider.

These tests exercise the evaluate_model → emit_metrics_json pipeline without
a GPU or real model weights. They catch orchestration bugs that unit tests
miss — wrong field names passed between layers, metric keys dropped, gate
logic not wired to the right metrics dict.
"""
import json
import pytest

from src.evaluation import harness
from src.evaluation.harness import evaluate_model, emit_metrics_json


SCHEMA = {
    "type": "object",
    "required": ["null_extraction"],
    "properties": {
        "null_extraction": {"type": "boolean"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type", "value"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "value": {"type": "string"},
                },
            },
        },
    },
}

EVAL_CFG = {
    "inference": {"provider": "hf_native"},
    "generation": {"max_new_tokens": 128, "temperature": 0.0, "do_sample": False},
    "constrained": {"enabled": False},
}

GATE_CFG = {
    "mode": "constrained",
    "fallback_to_raw": True,
    "thresholds": {"schema_validity": 0.90, "null_accuracy": 0.85, "field_f1": 0.50},
}

RAW_THRESHOLDS = {"schema_validity": 0.90, "null_accuracy": 0.85, "field_f1": 0.80}


class FakeProvider:
    """Returns preset predictions without a real model."""
    def __init__(self, predictions: list[str]):
        self._preds = predictions

    def generate(self, prompts: list[str], gen_cfg: dict) -> list[str]:
        return [self._preds[i % len(self._preds)] for i in range(len(prompts))]


def _examples(n_positive: int = 3, n_null: int = 2) -> list[dict]:
    pos_completion = json.dumps({
        "null_extraction": False,
        "entities": [{"name": "Apple", "type": "org", "value": "Apple"}],
    })
    null_completion = json.dumps({"null_extraction": True})
    examples = []
    for _ in range(n_positive):
        examples.append({"prompt": "p", "completion": pos_completion, "is_null_case": False})
    for _ in range(n_null):
        examples.append({"prompt": "p", "completion": null_completion, "is_null_case": True})
    return examples


class TestEvaluateModelEndToEnd:
    def test_perfect_predictions_score_max(self, monkeypatch):
        pos = json.dumps({"null_extraction": False, "entities": [{"name": "Apple", "type": "org", "value": "Apple"}]})
        null = json.dumps({"null_extraction": True})
        examples = _examples()

        def _fake_provider(name, path, cfg):
            # Return perfect predictions: pos for positives, null for nulls
            preds = [pos] * 3 + [null] * 2
            return FakeProvider(preds)

        monkeypatch.setattr(harness, "build_provider", _fake_provider)
        result = evaluate_model("test", "./fake", examples, SCHEMA, EVAL_CFG)

        assert result["model"] == "test"
        assert result["raw"]["schema_validity"] == 1.0
        assert result["raw"]["null_accuracy"] == 1.0
        assert result["n_examples"] == 5
        assert result["n_positive"] == 3
        assert result["n_null"] == 2

    def test_all_null_predictions_on_positive_examples(self, monkeypatch):
        # Model always abstains — null_accuracy for positive examples should be 0
        null = json.dumps({"null_extraction": True})
        monkeypatch.setattr(harness, "build_provider",
                            lambda *_: FakeProvider([null]))
        examples = _examples(n_positive=4, n_null=0)
        result = evaluate_model("test", "./fake", examples, SCHEMA, EVAL_CFG)

        assert result["raw"]["null_accuracy"] == 0.0   # all positive, all predicted null
        assert result["raw"]["field_f1"] == 0.0

    def test_schema_invalid_predictions(self, monkeypatch):
        monkeypatch.setattr(harness, "build_provider",
                            lambda *_: FakeProvider(["not json at all"]))
        result = evaluate_model("test", "./fake", _examples(), SCHEMA, EVAL_CFG)
        assert result["raw"]["schema_validity"] == 0.0

    def test_result_has_required_keys(self, monkeypatch):
        monkeypatch.setattr(harness, "build_provider",
                            lambda *_: FakeProvider(['{"null_extraction": true}']))
        result = evaluate_model("test", "./fake", _examples(), SCHEMA, EVAL_CFG)
        for key in ("model", "model_path", "provider", "raw", "constrained",
                    "raw_vs_guided_gap", "n_examples", "n_positive", "n_null"):
            assert key in result, f"Missing key: {key}"


class TestEndToEndPipeline:
    def test_evaluate_then_emit_produces_valid_metrics_json(self, monkeypatch, tmp_path):
        pos = json.dumps({"null_extraction": False, "entities": [{"name": "X", "type": "t", "value": "v"}]})
        null = json.dumps({"null_extraction": True})

        def _fake(name, path, cfg):
            preds = [pos, pos, null, pos, null]
            return FakeProvider(preds)

        monkeypatch.setattr(harness, "build_provider", _fake)

        examples = _examples(n_positive=3, n_null=2)
        result = evaluate_model("dpo", "./fake/dpo", examples, SCHEMA, EVAL_CFG)

        out_path = str(tmp_path / "metrics.json")
        output = emit_metrics_json(
            [result], RAW_THRESHOLDS, out_path,
            gate_cfg=GATE_CFG,
        )

        # File written
        assert (tmp_path / "metrics.json").exists()
        # Top-level structure
        assert "schema_version" in output
        assert "ci_gate_mode" in output
        assert "models" in output
        assert "ci_pass" in output
        # Model entry structure
        dpo = output["models"]["dpo"]
        assert "raw" in dpo
        assert "constrained" in dpo
        assert "pass_fail" in dpo
        assert "deployment_pass_fail" in dpo
        assert "deployment_gate_mode" in dpo

    def test_gate_falls_back_to_raw_when_constrained_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(harness, "build_provider",
                            lambda *_: FakeProvider(['{"null_extraction": true}']))
        examples = _examples()
        result = evaluate_model("sft", "./fake/sft", examples, SCHEMA, EVAL_CFG)

        output = emit_metrics_json(
            [result], RAW_THRESHOLDS, str(tmp_path / "m.json"),
            gate_cfg=GATE_CFG,   # mode=constrained but no constrained metrics
        )
        # constrained not run → falls back to raw
        assert output["models"]["sft"]["deployment_gate_mode"] == "raw"


class TestFixtureFile:
    def test_fixture_loads_correctly(self):
        from pathlib import Path
        fixture = Path("tests/fixtures/sample.jsonl")
        if not fixture.exists():
            pytest.skip("Run from project root")
        examples = [json.loads(l) for l in fixture.read_text().splitlines() if l.strip()]
        assert len(examples) == 5
        assert sum(1 for e in examples if e["is_null_case"]) == 2
        assert sum(1 for e in examples if not e["is_null_case"]) == 3
        for ex in examples:
            assert "prompt" in ex
            assert "completion" in ex
            assert "is_null_case" in ex
