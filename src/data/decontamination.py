"""
Dataset decontamination: hash-based and n-gram overlap deduplication
between train/val and the frozen test set.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _example_hash(example: dict[str, Any]) -> str:
    canonical = json.dumps(example, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _ngrams(text: str, n: int = 8) -> set[str]:
    tokens = text.lower().split()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def decontaminate(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    text_field: str = "input",
    ngram_n: int = 8,
    ngram_overlap_threshold: float = 0.5,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """
    Remove examples from train/val that appear in the frozen test set
    via exact hash match or n-gram overlap above threshold.

    Returns cleaned train, cleaned val, and a stats dict.
    """
    test_hashes = {_example_hash(ex) for ex in test}
    test_ngrams: set[str] = set()
    for ex in test:
        test_ngrams |= _ngrams(ex.get(text_field, ""), ngram_n)

    stats = {"hash_removed_train": 0, "ngram_removed_train": 0,
             "hash_removed_val": 0, "ngram_removed_val": 0}

    def is_contaminated(ex: dict) -> tuple[bool, str]:
        if _example_hash(ex) in test_hashes:
            return True, "hash"
        grams = _ngrams(ex.get(text_field, ""), ngram_n)
        if not grams:
            return False, ""
        overlap = len(grams & test_ngrams) / len(grams)
        if overlap >= ngram_overlap_threshold:
            return True, "ngram"
        return False, ""

    clean_train, clean_val = [], []
    for ex in train:
        contaminated, reason = is_contaminated(ex)
        if contaminated:
            stats[f"{reason}_removed_train"] += 1
        else:
            clean_train.append(ex)

    for ex in val:
        contaminated, reason = is_contaminated(ex)
        if contaminated:
            stats[f"{reason}_removed_val"] += 1
        else:
            clean_val.append(ex)

    return clean_train, clean_val, stats
