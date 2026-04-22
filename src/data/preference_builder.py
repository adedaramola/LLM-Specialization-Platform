"""
Preference dataset construction for DPO (Phase 2).

For each prompt, generates N completions from the SFT model at temperature > 0,
scores them via a deterministic rubric (schema validity + null-case correctness),
and emits chosen/rejected pairs. Null-case pairs are oversampled — DPO is
particularly effective at correcting over-eager extraction.
"""
from __future__ import annotations

import json
import random
from typing import Any

import jsonschema


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _parse_json_safe(text: str) -> tuple[dict | None, bool]:
    try:
        return json.loads(text.strip()), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def score_completion(
    completion_str: str,
    is_null_prompt: bool,
    schema: dict,
) -> float:
    """
    Deterministic rubric for JSON extraction completions. Returns 0.0–3.0.

    Points:
      +1.0  parseable JSON
      +1.0  passes schema validation
      +1.0  correct null/non-null prediction
    """
    obj, ok = _parse_json_safe(completion_str)
    if not ok:
        return 0.0

    try:
        jsonschema.validate(obj, schema)
        schema_score = 1.0
    except jsonschema.ValidationError:
        schema_score = 0.5  # parseable but schema-invalid

    pred_null = isinstance(obj, dict) and obj.get("null_extraction", False)
    null_score = 1.0 if (pred_null == is_null_prompt) else 0.0

    return 1.0 + schema_score + null_score  # always get +1 for parseable JSON


def build_preference_pairs(
    examples: list[dict],
    all_completions: list[list[str]],
    schema: dict,
) -> list[dict]:
    """
    Build (chosen, rejected) pairs from multiple completions per prompt.

    Pairs are only emitted when at least two completions have different scores.
    Null-case pairs are tagged for downstream oversampling validation.
    """
    pairs: list[dict] = []
    for example, completions in zip(examples, all_completions):
        is_null = example.get("is_null_case", False)
        scored = [
            (c, score_completion(c, is_null, schema))
            for c in completions
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_text, best_score = scored[0]
        # Pick the worst completion that is strictly worse than the best
        for worst_text, worst_score in reversed(scored):
            if worst_score < best_score:
                pairs.append({
                    "prompt": example["prompt"],
                    "chosen": best_text,
                    "rejected": worst_text,
                    "is_null_case": is_null,
                    "chosen_score": best_score,
                    "rejected_score": worst_score,
                })
                break

    return pairs


# ---------------------------------------------------------------------------
# Prompt sampling with null-case oversampling
# ---------------------------------------------------------------------------

def sample_prompts(
    examples: list[dict],
    target_pairs: int,
    null_case_fraction: float = 0.20,
    completions_per_prompt: int = 6,
    seed: int = 42,
) -> list[dict]:
    """
    Sample prompts from training examples, oversampling null cases.

    Assumes ~70% of prompts will yield a usable pair (some completions
    will be uniformly good/bad and produce no differential pair).
    """
    rng = random.Random(seed)
    needed = min(len(examples), int(target_pairs * 1.4))

    null_examples = [e for e in examples if e.get("is_null_case", False)]
    pos_examples = [e for e in examples if not e.get("is_null_case", False)]

    n_null = min(len(null_examples), int(needed * null_case_fraction))
    n_pos = min(len(pos_examples), needed - n_null)

    sampled = rng.sample(null_examples, n_null) + rng.sample(pos_examples, n_pos)
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_completions_batch(
    model,
    tokenizer,
    prompts: list[str],
    completions_per_prompt: int,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
) -> list[list[str]]:
    """
    Generate `completions_per_prompt` diverse outputs for each prompt.
    Uses temperature sampling (not greedy) to produce quality variation.
    """
    import torch

    results: list[list[str]] = []
    total = len(prompts)
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        completions: list[str] = []

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=completions_per_prompt,
                pad_token_id=tokenizer.eos_token_id,
            )

        for seq in output_ids:
            new_tokens = seq[prompt_len:]
            completions.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True)
            )
        results.append(completions)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Generated {i + 1}/{total} prompts", flush=True)

    return results


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_preference_pairs(pairs: list[dict], path: str) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def validate_preference_dataset(
    pairs: list[dict],
    target_pairs: int,
    null_case_fraction: float = 0.10,
) -> None:
    """Raise if the dataset is too small or null-case fraction is too low."""
    if len(pairs) < target_pairs * 0.5:
        raise ValueError(
            f"Only {len(pairs)} pairs generated; expected at least "
            f"{int(target_pairs * 0.5)}. Increase completions_per_prompt "
            "or lower target_pairs."
        )
    null_count = sum(1 for p in pairs if p.get("is_null_case", False))
    fraction = null_count / len(pairs) if pairs else 0.0
    if fraction < null_case_fraction:
        raise ValueError(
            f"Null-case pair fraction {fraction:.2%} below minimum "
            f"{null_case_fraction:.2%}. Add more null-case examples to "
            "training data or raise preference_data.null_case_fraction."
        )
