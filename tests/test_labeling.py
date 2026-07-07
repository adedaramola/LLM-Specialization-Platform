"""Tests for the v2 labeling contract validator."""
import pytest

from src.data.labeling import validate_labeled_output

TEXT = "Judge Patricia Hartwell ruled that Nexus Industries must divest within 180 days by 2025-03-15."


def _ent(name, etype, value):
    return {"name": name, "type": etype, "value": value}


def test_grounded_labels_pass():
    validate_labeled_output(
        {
            "null_extraction": False,
            "entities": [
                _ent("Patricia Hartwell", "person", "Patricia Hartwell"),
                _ent("Nexus Industries", "organization", "Nexus Industries"),
                _ent("180 days", "metric", "180 days"),
                _ent("2025-03-15", "date", "2025-03-15"),
            ],
        },
        TEXT,
    )


def test_invented_name_rejected():
    with pytest.raises(ValueError, match="not a verbatim substring"):
        validate_labeled_output(
            {"null_extraction": False, "entities": [_ent("Antitrust Ruling", "event", "Antitrust Ruling")]},
            TEXT,
        )


def test_descriptive_value_rejected():
    with pytest.raises(ValueError, match="must be identical to name"):
        validate_labeled_output(
            {"null_extraction": False, "entities": [_ent("Patricia Hartwell", "person", "Judge, District Court")]},
            TEXT,
        )


def test_unknown_type_rejected():
    with pytest.raises(ValueError, match="not in"):
        validate_labeled_output(
            {"null_extraction": False, "entities": [_ent("Nexus Industries", "product", "Nexus Industries")]},
            TEXT,
        )


def test_metric_value_must_be_verbatim():
    with pytest.raises(ValueError, match="not verbatim"):
        validate_labeled_output(
            {"null_extraction": False, "entities": [_ent("180 days", "metric", "6 months")]},
            TEXT,
        )


def test_date_iso_or_name():
    # ISO value allowed even if the surface form differs
    validate_labeled_output(
        {"null_extraction": False, "entities": [_ent("180 days", "date", "2025-03-15")]},
        TEXT,
    )
    with pytest.raises(ValueError, match="ISO-8601"):
        validate_labeled_output(
            {"null_extraction": False, "entities": [_ent("180 days", "date", "March 15th")]},
            TEXT,
        )


def test_null_with_entities_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        validate_labeled_output(
            {"null_extraction": True, "entities": [_ent("180 days", "metric", "180 days")]},
            TEXT,
        )


def test_positive_without_entities_rejected():
    with pytest.raises(ValueError, match="empty"):
        validate_labeled_output({"null_extraction": False, "entities": []}, TEXT)


def test_duplicate_entity_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_labeled_output(
            {
                "null_extraction": False,
                "entities": [
                    _ent("180 days", "metric", "180 days"),
                    _ent("180 days", "metric", "180 days"),
                ],
            },
            TEXT,
        )
