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
    reference_entities: list | None = None,
) -> float:
    """
    Deterministic rubric for JSON extraction completions. Returns 0.0–4.0.

    Points:
      +1.0  parseable JSON
      +1.0  passes schema validation (or +0.5 if schema-invalid)
      +1.0  correct null/non-null prediction
      +1.0  entity-level F1 against ground truth (continuous 0–1, positive cases only)

    The entity F1 component creates continuous score variation between completions
    that are otherwise all schema-valid and null-correct, enabling pair construction
    from a well-trained SFT model.
    """
    obj, ok = _parse_json_safe(completion_str)
    if not ok:
        return 0.0

    try:
        jsonschema.validate(obj, schema)
        schema_score = 1.0
    except jsonschema.ValidationError:
        schema_score = 0.5

    pred_null = isinstance(obj, dict) and obj.get("null_extraction", False)
    null_score = 1.0 if (pred_null == is_null_prompt) else 0.0

    base = 1.0 + schema_score + null_score

    # Entity-level F1 against ground truth for positive cases
    if reference_entities is not None and not is_null_prompt and null_score == 1.0:
        pred_names = {e.get("name", "").lower().strip() for e in obj.get("entities", [])}
        true_names = {e.get("name", "").lower().strip() for e in reference_entities if e.get("name")}
        if true_names:
            tp = len(pred_names & true_names)
            precision = tp / len(pred_names) if pred_names else 0.0
            recall = tp / len(true_names)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            f1 = 1.0
        return base + f1  # 0.0–4.0

    return base  # 0.0–3.0


def build_preference_pairs(
    examples: list[dict],
    all_completions: list[list[str]],
    schema: dict,
    seed: int = 42,
) -> list[dict]:
    """
    Build (chosen, rejected) pairs from multiple completions per prompt.

    For positive prompts: scores via entity-level F1 to create differentials.
    For null prompts where the SFT model is uniformly correct (all completions
    correctly abstain): synthesizes a pair using a real positive-case completion
    as the rejected response — a hallucinated extraction on a null prompt is the
    exact failure mode DPO is being used to suppress. These are marked
    synthetic_rejected=True in the output for transparency.
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    unpaired_null_cases: list[tuple[dict, str, float]] = []
    positive_completions: list[str] = []

    for example, completions in zip(examples, all_completions):
        is_null = example.get("is_null_case", False)

        ref_completion = example.get("completion", "")
        ref_obj, ref_ok = _parse_json_safe(ref_completion)
        reference_entities = ref_obj.get("entities", []) if ref_ok else None

        candidates = completions + ([ref_completion] if ref_ok else [])
        scored = [
            (c, score_completion(c, is_null, schema, reference_entities))
            for c in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_text, best_score = scored[0]
        pair_built = False
        for worst_text, worst_score in reversed(scored):
            if worst_score < best_score:
                pairs.append({
                    "prompt": example["prompt"],
                    "chosen": best_text,
                    "rejected": worst_text,
                    "is_null_case": is_null,
                    "chosen_score": best_score,
                    "rejected_score": worst_score,
                    "synthetic_rejected": False,
                })
                pair_built = True
                break

        # Null prompts where model is uniformly correct produce no score differential.
        # Queue them for synthetic pair construction after the main loop, once we have
        # a pool of real positive-case completions to use as realistic rejected responses.
        if not pair_built and is_null and best_score >= 2.0:
            unpaired_null_cases.append((example, best_text, best_score))

        # Collect schema-valid, non-null extractions from positive prompts as a
        # rejection pool for the synthetic null pairs above.
        if not is_null:
            for c in completions:
                obj, ok = _parse_json_safe(c)
                if ok and not obj.get("null_extraction", False):
                    try:
                        jsonschema.validate(obj, schema)
                        positive_completions.append(c)
                    except jsonschema.ValidationError:
                        pass

    # Build synthetic null-case pairs: chosen=correct abstention, rejected=hallucinated extraction.
    # Only emit when there are real positive-case completions to use as rejected responses —
    # a fabricated rejected response with no grounding in the model's actual output adds noise.
    for example, chosen_text, chosen_score in unpaired_null_cases:
        if not positive_completions:
            continue
        rejected_text = rng.choice(positive_completions)
        pairs.append({
            "prompt": example["prompt"],
            "chosen": chosen_text,
            "rejected": rejected_text,
            "is_null_case": True,
            "chosen_score": chosen_score,
            "rejected_score": 2.0,
            "synthetic_rejected": True,
        })

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
    batch_size: int = 4,
) -> list[list[str]]:
    """
    Generate `completions_per_prompt` diverse outputs for each prompt.

    Each prompt is repeated `completions_per_prompt` times in the batch so
    temperature sampling produces diverse outputs with num_return_sequences=1.
    This avoids HF's known limitation with num_return_sequences > 1 + batch > 1.
    Prompts are processed in groups of `batch_size` (GPU batch = batch_size *
    completions_per_prompt sequences).
    """
    import torch

    # Left-padding required for batched generation with decoder-only models
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results: list[list[str]] = []
    total = len(prompts)

    for batch_start in range(0, total, batch_size):
        batch_prompts = prompts[batch_start: batch_start + batch_size]

        # Expand: [p0, p1, p2, p3] → [p0,p0,...,p0, p1,p1,...,p1, ...]
        expanded = [p for p in batch_prompts for _ in range(completions_per_prompt)]

        inputs = tokenizer(
            expanded,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        padded_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # output_ids: [batch_size * completions_per_prompt, padded_len + new_tokens]
        for b in range(len(batch_prompts)):
            completions: list[str] = []
            for n in range(completions_per_prompt):
                seq = output_ids[b * completions_per_prompt + n]
                new_tokens = seq[padded_len:]
                completions.append(
                    tokenizer.decode(new_tokens, skip_special_tokens=True)
                )
            results.append(completions)

        done = min(batch_start + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"  Generated {done}/{total} prompts", flush=True)

    tokenizer.padding_side = original_padding_side
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
    if len(pairs) < target_pairs * 0.3:
        raise ValueError(
            f"Only {len(pairs)} pairs generated; expected at least "
            f"{int(target_pairs * 0.5)}. Increase completions_per_prompt "
            "or lower target_pairs."
        )
    null_count = sum(1 for p in pairs if p.get("is_null_case", False))
    fraction = null_count / len(pairs) if pairs else 0.0
    if null_case_fraction > 0 and fraction < null_case_fraction:
        raise ValueError(
            f"Null-case pair fraction {fraction:.2%} below target "
            f"{null_case_fraction:.2%}. Add more null-case prompts or lower null_case_fraction."
        )
