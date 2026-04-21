"""Tests for dataset preparation and decontamination."""
import json
import pytest

from src.data.decontamination import decontaminate
from src.data.dataset_builder import (
    format_example,
    split_dataset,
    validate_null_fraction,
    content_hash_dataset,
)


def _make_example(text: str, null: bool = False) -> dict:
    return {
        "input": text,
        "output": {"null_extraction": null} if null else {
            "null_extraction": False,
            "entities": [{"name": "x", "type": "t", "value": text[:10]}],
        },
    }


class TestSplitDataset:
    def test_sizes(self):
        examples = [_make_example(f"text {i}") for i in range(100)]
        train, val, test = split_dataset(examples, (0.8, 0.1, 0.1), seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_reproducible(self):
        examples = [_make_example(f"text {i}") for i in range(50)]
        t1, v1, te1 = split_dataset(examples, seed=42)
        t2, v2, te2 = split_dataset(examples, seed=42)
        assert [e["input"] for e in t1] == [e["input"] for e in t2]


class TestValidateNullFraction:
    def test_passes_when_sufficient(self):
        examples = [{"is_null_case": True}] * 15 + [{"is_null_case": False}] * 85
        validate_null_fraction(examples, min_fraction=0.10)

    def test_raises_when_insufficient(self):
        examples = [{"is_null_case": True}] * 5 + [{"is_null_case": False}] * 95
        with pytest.raises(ValueError, match="Null-case fraction"):
            validate_null_fraction(examples, min_fraction=0.10)


class TestDecontamination:
    def test_removes_exact_duplicates(self):
        test = [{"input": "secret text", "output": {}}]
        train = [{"input": "secret text", "output": {}}, {"input": "other text", "output": {}}]
        val: list = []
        clean_train, clean_val, stats = decontaminate(train, val, test)
        assert len(clean_train) == 1
        assert clean_train[0]["input"] == "other text"
        assert stats["hash_removed_train"] == 1

    def test_preserves_clean_examples(self):
        test = [{"input": "test text", "output": {}}]
        train = [{"input": "totally different content here", "output": {}}]
        clean_train, _, _ = decontaminate(train, [], test)
        assert len(clean_train) == 1


class TestContentHash:
    def test_deterministic(self):
        examples = [{"a": 1}, {"b": 2}]
        assert content_hash_dataset(examples) == content_hash_dataset(examples)

    def test_order_sensitive(self):
        e1, e2 = {"a": 1}, {"b": 2}
        assert content_hash_dataset([e1, e2]) != content_hash_dataset([e2, e1])
