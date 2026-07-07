"""
Labeling contract for the JSON extraction task (v2).

v1 labels left "name" and "value" undefined, so the generating model invented
a different convention per sample (~42% of entity names were descriptive
labels not present in the text, ~35% of values were not verbatim). Exact-match
field F1 caps at whatever fraction of labels happens to be guessable — that
label noise, not model capacity, was the v1 score ceiling.

v2 makes every label deterministic given the input text:
  - name  = verbatim span from the input
  - value = name, except dates (ISO-8601 when fully stated) and metrics
            (quantity+unit verbatim)
and enforces it programmatically: any entity that cannot be grounded in the
input text is a validation error, not a warning.
"""
from __future__ import annotations

import re
from typing import Any

PROMPT_VERSION = "2.0.0"

ALLOWED_TYPES = {"person", "organization", "location", "date", "event", "metric"}

_ISO_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")

LABELING_RULES = """Label entities using EXACTLY these rules — they are enforced by a validator and violations are rejected:

1. "name" MUST be a verbatim contiguous substring of the input text, copied exactly (same casing, punctuation, spacing). Never invent, paraphrase, or describe.
2. "type" MUST be one of: person, organization, location, date, event, metric.
3. "value" rules by type:
   - person, organization, location, event: "value" MUST be identical to "name".
   - date: if the input states a complete calendar date (day, month, and year), "value" is that date in ISO-8601 (YYYY-MM-DD). Otherwise "value" MUST be identical to "name" (e.g. name "2035" -> value "2035"; name "Tuesday" -> value "Tuesday").
   - metric: "name" and "value" are BOTH the quantity with its unit exactly as written in the text (e.g. "$42 million", "4.2%", "180 days", "12,000 jobs"). Do not convert units or reformat numbers.
4. person: the full name as written at first mention. organization: the name as written — if the text says "UN", use "UN", never "United Nations".
5. event: ONLY when the text contains a proper-noun event name (e.g. "Global Climate Accord Summit"). Never invent a descriptive title for something the text merely describes.
6. Extract each distinct real-world entity ONCE, using its first-mention surface form. Do not list the same (name, type) twice.
7. Every entity must be explicitly present in the text. Nothing inferred from context.
8. null_extraction is true only when the text contains no extractable entities at all; then "entities" MUST be [].
"""


def validate_labeled_output(output: dict[str, Any], input_text: str) -> None:
    """Raise ValueError describing every contract violation in `output`."""
    errors: list[str] = []

    if not isinstance(output.get("null_extraction"), bool):
        raise ValueError("output.null_extraction must be a boolean")
    entities = output.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("output.entities must be a list")

    if output["null_extraction"]:
        if entities:
            errors.append("null_extraction=true but entities is non-empty")
    elif not entities:
        errors.append("null_extraction=false but entities is empty")

    seen: set[tuple[str, str]] = set()
    for i, ent in enumerate(entities):
        if not isinstance(ent, dict) or not {"name", "type", "value"} <= ent.keys():
            errors.append(f"entity {i}: missing name/type/value")
            continue
        name, etype, value = ent["name"], ent["type"], ent["value"]

        if etype not in ALLOWED_TYPES:
            errors.append(f"entity {i} ({name!r}): type {etype!r} not in {sorted(ALLOWED_TYPES)}")
        if not isinstance(name, str) or name not in input_text:
            errors.append(f"entity {i}: name {name!r} is not a verbatim substring of the input")
        if (name, etype) in seen:
            errors.append(f"entity {i}: duplicate (name, type) ({name!r}, {etype!r})")
        seen.add((name, etype))

        if etype == "date":
            if value != name and not (isinstance(value, str) and _ISO_DATE_RE.match(value)):
                errors.append(
                    f"entity {i} ({name!r}): date value {value!r} must equal name or be ISO-8601"
                )
        elif etype == "metric":
            if not isinstance(value, str) or value not in input_text:
                errors.append(
                    f"entity {i} ({name!r}): metric value {value!r} is not verbatim in the input"
                )
        else:
            if value != name:
                errors.append(
                    f"entity {i} ({name!r}): value {value!r} must be identical to name for type {etype}"
                )

    if errors:
        raise ValueError("; ".join(errors))
