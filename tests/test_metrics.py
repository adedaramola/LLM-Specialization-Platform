"""Tests for evaluation metrics."""
import json
import pytest

from src.evaluation.metrics import (
    schema_validity,
    null_accuracy,
    field_level_f1,
    exact_match,
    compute_all_metrics,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "null_extraction": {"type": "boolean"},
        "entities": {"type": "array"},
    },
    "required": ["null_extraction"],
}


class TestSchemaValidity:
    def test_valid_null(self):
        preds = ['{"null_extraction": true}']
        assert schema_validity(preds, SCHEMA) == 1.0

    def test_invalid_json(self):
        preds = ["not json"]
        assert schema_validity(preds, SCHEMA) == 0.0

    def test_empty(self):
        assert schema_validity([], SCHEMA) == 0.0


class TestNullAccuracy:
    def test_correct_abstention(self):
        preds = ['{"null_extraction": true}']
        labels = [True]
        result = null_accuracy(preds, labels)
        assert result["null_accuracy"] == 1.0

    def test_hallucinated_extraction(self):
        preds = ['{"null_extraction": false, "entities": [{"name": "x", "type": "y", "value": "z"}]}']
        labels = [True]
        result = null_accuracy(preds, labels)
        assert result["null_accuracy"] == 0.0
        assert result["fp"] == 1

    def test_correct_extraction(self):
        preds = ['{"null_extraction": false, "entities": [{"name": "x", "type": "y", "value": "z"}]}']
        labels = [False]
        result = null_accuracy(preds, labels)
        assert result["tp"] == 1


class TestFieldLevelF1:
    def test_exact_match(self):
        pred = json.dumps({"null_extraction": False, "entities": [{"name": "a", "type": "t", "value": "v"}]})
        ref = pred
        result = field_level_f1([pred], [ref])
        assert result["field_f1"] == 1.0

    def test_null_case_excluded_when_labeled(self):
        # Null cases must be excluded from field F1 — abstention quality is in null_accuracy
        pred = ref = '{"null_extraction": true}'
        result = field_level_f1([pred], [ref], null_labels=[True])
        assert result["field_f1"] == 0.0  # n=0, no positive examples

    def test_null_case_without_labels_scores_zero(self):
        # Without labels, both-empty is a degenerate positive case: no entities extracted = 0 F1
        pred = ref = '{"null_extraction": true}'
        result = field_level_f1([pred], [ref])
        assert result["field_f1"] == 0.0

    def test_over_abstain_scores_zero(self):
        # A model predicting null on a positive example gets field_f1=0
        pred = '{"null_extraction": true}'
        ref = json.dumps({"null_extraction": False, "entities": [{"name": "a", "type": "t", "value": "v"}]})
        result = field_level_f1([pred], [ref], null_labels=[False])
        assert result["field_f1"] == 0.0


class TestExactMatch:
    def test_exact(self):
        s = '{"null_extraction": true}'
        assert exact_match([s], [s]) == 1.0

    def test_mismatch(self):
        assert exact_match(['{"null_extraction": true}'], ['{"null_extraction": false}']) == 0.0


class TestMetricRobustness:
    def test_garbage_is_not_abstention(self):
        # Unparsable output on a null case must NOT count as correct abstention
        result = null_accuracy(["not json at all"], [True])
        assert result["null_accuracy"] == 0.0
        assert result["fp"] == 1

    def test_double_parse_failure_is_not_exact_match(self):
        assert exact_match(["garbage"], ["also garbage"]) == 0.0

    def test_field_f1_normalizes_case_and_whitespace(self):
        pred = json.dumps({"null_extraction": False, "entities": [{"name": "acme  corp", "type": "organization", "value": "ACME Corp"}]})
        ref = json.dumps({"null_extraction": False, "entities": [{"name": "Acme Corp", "type": "organization", "value": "acme corp"}]})
        assert field_level_f1([pred], [ref], [False])["field_f1"] == 1.0

    def test_field_f1_duplicate_names_not_collapsed(self):
        # Two identical reference entities: predicting only one is recall 0.5, not 1.0
        ent = {"name": "a", "type": "t", "value": "v"}
        ref = json.dumps({"null_extraction": False, "entities": [ent, ent]})
        pred = json.dumps({"null_extraction": False, "entities": [ent]})
        result = field_level_f1([pred], [ref], [False])
        assert result["field_recall"] == 0.5
        assert result["field_precision"] == 1.0

    def test_field_f1_unparsable_reference_excluded(self):
        good = json.dumps({"null_extraction": False, "entities": [{"name": "a", "type": "t", "value": "v"}]})
        # One good pair + one broken reference: average over the 1 valid pair
        result = field_level_f1([good, good], [good, "broken ref"], [False, False])
        assert result["field_f1"] == 1.0
