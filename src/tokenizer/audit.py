"""
Phase 0 tokenizer audit: verify task-critical characters are represented
faithfully, detect byte-level fallbacks, document chat-template modifications.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


TASK_CRITICAL_CHARS = [
    ("{", "open_brace"),
    ("}", "close_brace"),
    ("[", "open_bracket"),
    ("]", "close_bracket"),
    ('"', "double_quote"),
    ("'", "single_quote"),
    (",", "comma"),
    (":", "colon"),
    ("\\", "backslash"),
    ("\n", "newline"),
    ("\t", "tab"),
    ("  ", "two_spaces"),
    ("null", "null_keyword"),
    ("true", "true_keyword"),
    ("false", "false_keyword"),
]

STRUCTURED_PROBES = [
    '{"key": "value"}',
    '{"null_extraction": true}',
    '{"null_extraction": false, "entities": [{"name": "x", "type": "y", "value": null}]}',
    '[1, 2, 3]',
    '{"a": {"b": {"c": "deep"}}}',
]


@dataclass
class TokenizerAuditReport:
    model_name: str
    vocab_size: int
    has_bos: bool
    has_eos: bool
    chat_template_present: bool
    added_tokens: list[str]
    char_coverage: dict[str, dict[str, Any]]
    byte_fallback_chars: list[str]
    probe_roundtrips: list[dict[str, Any]]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "has_bos": self.has_bos,
            "has_eos": self.has_eos,
            "chat_template_present": self.chat_template_present,
            "added_tokens": self.added_tokens,
            "char_coverage": self.char_coverage,
            "byte_fallback_chars": self.byte_fallback_chars,
            "probe_roundtrips": self.probe_roundtrips,
            "issues": self.issues,
            "warnings": self.warnings,
            "passed": len(self.issues) == 0,
        }


def _is_byte_fallback(token_str: str) -> bool:
    return token_str.startswith("<0x") and token_str.endswith(">")


def audit_tokenizer(model_name_or_path: str) -> TokenizerAuditReport:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

    issues: list[str] = []
    warnings: list[str] = []

    added_tokens = [t for t in tok.additional_special_tokens if t not in ["<s>", "</s>"]]

    char_coverage: dict[str, dict[str, Any]] = {}
    byte_fallback_chars: list[str] = []

    for char, label in TASK_CRITICAL_CHARS:
        ids = tok.encode(char, add_special_tokens=False)
        tokens = [tok.convert_ids_to_tokens(i) for i in ids]
        is_single = len(ids) == 1
        has_byte_fb = any(_is_byte_fallback(str(t)) for t in tokens)

        char_coverage[label] = {
            "char": repr(char),
            "token_ids": ids,
            "tokens": [str(t) for t in tokens],
            "single_token": is_single,
            "byte_fallback": has_byte_fb,
        }

        if has_byte_fb:
            byte_fallback_chars.append(label)
            warnings.append(
                f"Character '{label}' ({repr(char)}) falls back to byte tokens. "
                "This increases sequence length and may hurt structured-output accuracy."
            )

    probe_roundtrips: list[dict[str, Any]] = []
    for probe in STRUCTURED_PROBES:
        ids = tok.encode(probe, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        exact = decoded == probe
        probe_roundtrips.append({
            "probe": probe,
            "token_count": len(ids),
            "decoded": decoded,
            "exact_roundtrip": exact,
        })
        if not exact:
            issues.append(f"Roundtrip failure for probe: {probe!r} → {decoded!r}")

    chat_template_present = tok.chat_template is not None
    if not chat_template_present:
        warnings.append(
            "No chat_template found on tokenizer. "
            "Chat formatting must be applied manually and kept consistent across "
            "adapter, merged weights, and GGUF."
        )

    return TokenizerAuditReport(
        model_name=model_name_or_path,
        vocab_size=tok.vocab_size,
        has_bos=tok.bos_token is not None,
        has_eos=tok.eos_token is not None,
        chat_template_present=chat_template_present,
        added_tokens=added_tokens,
        char_coverage=char_coverage,
        byte_fallback_chars=byte_fallback_chars,
        probe_roundtrips=probe_roundtrips,
        issues=issues,
        warnings=warnings,
    )


def save_audit_report(report: TokenizerAuditReport, path: str) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
