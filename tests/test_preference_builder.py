"""Tests for preference dataset construction."""
import json
import pytest

from src.data.preference_builder import (
    score_completion,
    build_preference_pairs,
    sample_prompts,
    validate_preference_dataset,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "null_extraction": {"type": "boolean"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "value": {"type": ["string", "number", "boolean", "null"]},
                },
                "required": ["name", "type", "value"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["null_extraction"],
    "if": {"properties": {"null_extraction": {"const": False}}},
    "then": {"required": ["entities"], "properties": {"entities": {"minItems": 1}}},
    "additionalProperties": False,
}

GOOD_NULL = json.dumps({"null_extraction": True})
GOOD_EXTRACTION = json.dumps({
    "null_extraction": False,
    "entities": [{"name": "x", "type": "string", "value": "v"}],
})
BAD_JSON = "not json at all"
SCHEMA_INVALID = json.dumps({"wrong_field": True})


class TestScoreCompletion:
    def test_good_null_on_null_prompt(self):
        score = score_completion(GOOD_NULL, is_null_prompt=True, schema=SCHEMA)
        assert score == 3.0

    def test_good_extraction_on_pos_prompt(self):
        score = score_completion(GOOD_EXTRACTION, is_null_prompt=False, schema=SCHEMA)
        assert score == 3.0

    def test_hallucination_penalised(self):
        # Non-null output on a null prompt — wrong null prediction
        score = score_completion(GOOD_EXTRACTION, is_null_prompt=True, schema=SCHEMA)
        assert score < 3.0

    def test_bad_json_scores_zero(self):
        assert score_completion(BAD_JSON, is_null_prompt=False, schema=SCHEMA) == 0.0

    def test_schema_invalid_partial_score(self):
        score = score_completion(SCHEMA_INVALID, is_null_prompt=False, schema=SCHEMA)
        assert 0.0 < score < 3.0


class TestBuildPreferencePairs:
    def _make_example(self, is_null: bool = False) -> dict:
        return {"prompt": "test prompt", "is_null_case": is_null}

    def test_emits_pair_when_scores_differ(self):
        examples = [self._make_example(is_null=True)]
        completions = [[GOOD_NULL, BAD_JSON]]  # score 3.0 vs 0.0
        pairs = build_preference_pairs(examples, completions, SCHEMA)
        assert len(pairs) == 1
        assert pairs[0]["chosen"] == GOOD_NULL
        assert pairs[0]["rejected"] == BAD_JSON

    def test_no_pair_when_all_equal(self):
        examples = [self._make_example(is_null=True)]
        completions = [[GOOD_NULL, GOOD_NULL]]  # identical scores
        pairs = build_preference_pairs(examples, completions, SCHEMA)
        assert len(pairs) == 0

    def test_null_case_tag_propagated(self):
        examples = [self._make_example(is_null=True)]
        completions = [[GOOD_NULL, BAD_JSON]]
        pairs = build_preference_pairs(examples, completions, SCHEMA)
        assert pairs[0]["is_null_case"] is True

    def test_chosen_score_gt_rejected_score(self):
        examples = [self._make_example(is_null=False)]
        completions = [[GOOD_EXTRACTION, BAD_JSON, SCHEMA_INVALID]]
        pairs = build_preference_pairs(examples, completions, SCHEMA)
        assert len(pairs) == 1
        assert pairs[0]["chosen_score"] > pairs[0]["rejected_score"]


class TestSamplePrompts:
    def _make_examples(self, n_pos: int, n_null: int) -> list[dict]:
        pos = [{"prompt": f"p{i}", "is_null_case": False} for i in range(n_pos)]
        null = [{"prompt": f"n{i}", "is_null_case": True} for i in range(n_null)]
        return pos + null

    def test_respects_null_fraction(self):
        examples = self._make_examples(n_pos=200, n_null=50)
        sampled = sample_prompts(examples, target_pairs=100, null_case_fraction=0.20)
        null_frac = sum(1 for e in sampled if e["is_null_case"]) / len(sampled)
        assert null_frac >= 0.15  # some tolerance

    def test_reproducible(self):
        examples = self._make_examples(100, 20)
        s1 = sample_prompts(examples, 50, seed=42)
        s2 = sample_prompts(examples, 50, seed=42)
        assert [e["prompt"] for e in s1] == [e["prompt"] for e in s2]

    def test_does_not_exceed_available(self):
        examples = self._make_examples(5, 2)
        sampled = sample_prompts(examples, target_pairs=10000)
        assert len(sampled) <= len(examples)


class TestValidatePreferenceDataset:
    def _make_pairs(self, n: int, null_frac: float = 0.20) -> list[dict]:
        pairs = []
        for i in range(n):
            pairs.append({
                "prompt": f"p{i}",
                "chosen": GOOD_NULL,
                "rejected": BAD_JSON,
                "is_null_case": i < int(n * null_frac),
                "chosen_score": 3.0,
                "rejected_score": 0.0,
            })
        return pairs

    def test_passes_valid_dataset(self):
        pairs = self._make_pairs(200)
        validate_preference_dataset(pairs, target_pairs=200)

    def test_raises_too_few_pairs(self):
        pairs = self._make_pairs(10)
        with pytest.raises(ValueError, match="pairs generated"):
            validate_preference_dataset(pairs, target_pairs=200)

    def test_raises_insufficient_null_fraction(self):
        pairs = self._make_pairs(200, null_frac=0.01)
        with pytest.raises(ValueError, match="Null-case pair fraction"):
            validate_preference_dataset(pairs, target_pairs=200, null_case_fraction=0.10)
