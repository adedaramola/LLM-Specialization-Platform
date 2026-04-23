"""
Task metrics: schema validity, field-level P/R/F1, exact match,
type correctness, null-extraction accuracy, tool-calling metrics.
"""
from __future__ import annotations

import json
from typing import Any

import jsonschema


def _parse_json_safe(text: str) -> tuple[dict | None, bool]:
    try:
        return json.loads(text.strip()), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def schema_validity(predictions: list[str], schema: dict) -> float:
    valid = 0
    for pred in predictions:
        obj, ok = _parse_json_safe(pred)
        if not ok:
            continue
        try:
            jsonschema.validate(obj, schema)
            valid += 1
        except jsonschema.ValidationError:
            pass
    return valid / len(predictions) if predictions else 0.0


def null_accuracy(
    predictions: list[str], labels: list[bool]
) -> dict[str, float]:
    assert len(predictions) == len(labels)
    tp = tn = fp = fn = 0
    for pred_str, is_null in zip(predictions, labels):
        obj, ok = _parse_json_safe(pred_str)
        pred_null = (not ok) or (isinstance(obj, dict) and obj.get("null_extraction", False))
        if is_null and pred_null:
            tn += 1  # correctly abstained
        elif not is_null and not pred_null:
            tp += 1  # correctly extracted
        elif is_null and not pred_null:
            fp += 1  # hallucinated extraction
        else:
            fn += 1  # missed extraction

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    null_acc = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "null_accuracy": null_acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def field_level_f1(
    predictions: list[str],
    references: list[str],
    null_labels: list[bool] | None = None,
) -> dict[str, float]:
    """Field-level P/R/F1 computed on positive examples only.

    Null cases are excluded: abstention correctness is captured by null_accuracy,
    not by field extraction quality. Including null cases in this metric inflates
    it to exactly the null-case fraction for any model that over-abstains.
    """
    total_p = total_r = total_f1 = 0.0
    n = 0
    for i, (pred_str, ref_str) in enumerate(zip(predictions, references)):
        if null_labels is not None and null_labels[i]:
            continue  # null cases are measured by null_accuracy, not here

        pred_obj, pred_ok = _parse_json_safe(pred_str)
        ref_obj, ref_ok = _parse_json_safe(ref_str)
        if not ref_ok:
            n += 1
            continue

        pred_entities = {
            e["name"]: (e["type"], e["value"])
            for e in (pred_obj or {}).get("entities", [])
            if isinstance(e, dict) and "name" in e
        } if pred_ok else {}
        ref_entities = {
            e["name"]: (e["type"], e["value"])
            for e in ref_obj.get("entities", [])
            if isinstance(e, dict) and "name" in e
        }

        correct = sum(
            1 for k, v in pred_entities.items()
            if k in ref_entities and ref_entities[k] == v
        )
        p = correct / len(pred_entities) if pred_entities else 0.0
        r = correct / len(ref_entities) if ref_entities else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        total_p += p
        total_r += r
        total_f1 += f1
        n += 1

    if n == 0:
        return {"field_precision": 0.0, "field_recall": 0.0, "field_f1": 0.0}
    return {
        "field_precision": total_p / n,
        "field_recall": total_r / n,
        "field_f1": total_f1 / n,
    }


def exact_match(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(
        _parse_json_safe(p)[0] == _parse_json_safe(r)[0]
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions)


def tool_calling_metrics(
    predictions: list[dict], references: list[dict]
) -> dict[str, float]:
    tool_sel = arg_name = arg_value = call_valid = 0
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        if pred.get("tool") == ref.get("tool"):
            tool_sel += 1
        pred_args = pred.get("arguments", {})
        ref_args = ref.get("arguments", {})
        if ref_args:
            name_match = set(pred_args.keys()) == set(ref_args.keys())
            val_match = all(pred_args.get(k) == v for k, v in ref_args.items())
        else:
            name_match = val_match = not pred_args
        if name_match:
            arg_name += 1
        if val_match:
            arg_value += 1
        if pred.get("tool") == ref.get("tool") and name_match and val_match:
            call_valid += 1

    return {
        "tool_selection_accuracy": tool_sel / n if n else 0.0,
        "arg_name_accuracy": arg_name / n if n else 0.0,
        "arg_value_accuracy": arg_value / n if n else 0.0,
        "call_validity": call_valid / n if n else 0.0,
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    null_labels: list[bool],
    schema: dict,
) -> dict[str, float]:
    result: dict[str, float] = {}
    result["schema_validity"] = schema_validity(predictions, schema)
    result["exact_match"] = exact_match(predictions, references)
    result.update(field_level_f1(predictions, references, null_labels))
    result.update(null_accuracy(predictions, null_labels))
    return result
