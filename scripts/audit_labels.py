"""
Label-grounding audit: measures how much of the dataset is learnable.

Field F1 requires the model to reproduce the reference entity strings exactly.
If an entity's name/value cannot be derived from the input text, the label is
noise and permanently caps the score. This audit quantifies that:

  - name grounding:  entity name appears verbatim in the input text
  - value grounding: value verbatim in text, equal to name, or ISO-8601 date

Run on raw JSONL ({"input", "output"}) or prepared JSONL ({"prompt",
"completion", "is_null_case"}).

Usage:
  python scripts/audit_labels.py data/raw/news_extraction_v2.jsonl [--min-grounding 0.95]

Exits non-zero when grounding falls below --min-grounding, so it can gate
dataset preparation in CI.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labeling import ALLOWED_TYPES, _ISO_DATE_RE


def _extract_fields(row: dict) -> tuple[str, dict, bool]:
    """Return (input_text, output_obj, is_null) for raw or prepared rows."""
    if "input" in row:
        out = row["output"]
        return row["input"], out, out.get("null_extraction", False)
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", row["prompt"], re.S)
    return m.group(1) if m else "", json.loads(row["completion"]), row.get("is_null_case", False)


def audit(path: str) -> dict:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    n_pos = n_null = n_entities = 0
    name_grounded = value_grounded = bad_type = dup_names = 0
    type_counter: collections.Counter = collections.Counter()

    for row in rows:
        text, output, is_null = _extract_fields(row)
        if is_null:
            n_null += 1
            continue
        n_pos += 1
        entities = output.get("entities", [])
        names = [e.get("name") for e in entities]
        if len(names) != len(set(names)):
            dup_names += 1
        for e in entities:
            n_entities += 1
            name, etype, value = e.get("name", ""), e.get("type", ""), e.get("value", "")
            type_counter[etype] += 1
            if etype not in ALLOWED_TYPES:
                bad_type += 1
            if isinstance(name, str) and name in text:
                name_grounded += 1
            if (
                value == name
                or (isinstance(value, str) and value in text)
                or (etype == "date" and isinstance(value, str) and _ISO_DATE_RE.match(value))
            ):
                value_grounded += 1

    return {
        "path": path,
        "rows": len(rows),
        "positive": n_pos,
        "null": n_null,
        "null_fraction": n_null / len(rows) if rows else 0.0,
        "entities": n_entities,
        "entities_per_positive": n_entities / n_pos if n_pos else 0.0,
        "name_grounding": name_grounded / n_entities if n_entities else 1.0,
        "value_grounding": value_grounded / n_entities if n_entities else 1.0,
        "invalid_type_count": bad_type,
        "duplicate_name_examples": dup_names,
        "type_distribution": dict(type_counter.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit label grounding in a dataset")
    parser.add_argument("paths", nargs="+", help="JSONL files (raw or prepared format)")
    parser.add_argument("--min-grounding", type=float, default=0.95,
                        help="Fail when name or value grounding is below this (default 0.95)")
    args = parser.parse_args()

    failed = False
    for path in args.paths:
        stats = audit(path)
        print(json.dumps(stats, indent=2))
        if stats["name_grounding"] < args.min_grounding or stats["value_grounding"] < args.min_grounding:
            print(f"FAIL: grounding below {args.min_grounding:.0%} in {path} — "
                  "these labels are unlearnable and will cap field F1.", file=sys.stderr)
            failed = True
        if stats["invalid_type_count"]:
            print(f"FAIL: {stats['invalid_type_count']} entities with types outside "
                  f"{sorted(ALLOWED_TYPES)} in {path}", file=sys.stderr)
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
