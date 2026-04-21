"""
General-capability regression check.
Small MMLU/HellaSwag slice to catch catastrophic forgetting from DPO.
"""
from __future__ import annotations

import json
import random
from typing import Any


def _load_mmlu_tiny(subjects: list[str], n_per_subject: int = 35) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    examples = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        indices = random.sample(range(len(ds)), min(n_per_subject, len(ds)))
        for i in indices:
            row = ds[i]
            choices = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(row["choices"]))
            examples.append({
                "benchmark": "mmlu",
                "subject": subject,
                "prompt": f"Question: {row['question']}\n{choices}\nAnswer:",
                "answer": chr(65 + row["answer"]),
            })
    return examples


def _load_hellaswag_tiny(n: int = 100) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    ds = load_dataset("Rowan/hellaswag", split="validation")
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    examples = []
    for i in indices:
        row = ds[i]
        endings = "\n".join(f"{j}. {e}" for j, e in enumerate(row["endings"]))
        examples.append({
            "benchmark": "hellaswag",
            "prompt": f"{row['ctx']}\n{endings}\nBest continuation (number):",
            "answer": str(row["label"]),
        })
    return examples


def run_regression(
    provider,
    gen_cfg: dict[str, Any],
    benchmarks: list[str],
    mmlu_subjects: list[str],
    num_samples: int,
    seed: int = 42,
) -> dict[str, float]:
    random.seed(seed)
    results: dict[str, float] = {}

    if "mmlu_tiny" in benchmarks:
        examples = _load_mmlu_tiny(mmlu_subjects, n_per_subject=max(1, num_samples // len(mmlu_subjects)))
        if examples:
            prompts = [ex["prompt"] for ex in examples]
            preds = provider.generate(prompts, {**gen_cfg, "max_new_tokens": 5})
            correct = sum(
                pred.strip().upper().startswith(ex["answer"])
                for pred, ex in zip(preds, examples)
            )
            results["mmlu_accuracy"] = correct / len(examples)

    if "hellaswag_tiny" in benchmarks:
        examples = _load_hellaswag_tiny(num_samples)
        if examples:
            prompts = [ex["prompt"] for ex in examples]
            preds = provider.generate(prompts, {**gen_cfg, "max_new_tokens": 5})
            correct = sum(
                pred.strip().startswith(ex["answer"])
                for pred, ex in zip(preds, examples)
            )
            results["hellaswag_accuracy"] = correct / len(examples)

    return results


def check_regression(
    base_results: dict[str, float],
    new_results: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    report: dict[str, Any] = {"passed": True, "deltas": {}}
    for key, new_val in new_results.items():
        base_val = base_results.get(key, 0.0)
        delta = new_val - base_val
        threshold_key = key.replace("_accuracy", "") + "_delta"
        min_delta = thresholds.get(threshold_key, -0.05)
        passed = delta >= min_delta
        report["deltas"][key] = {"base": base_val, "new": new_val, "delta": delta, "passed": passed}
        if not passed:
            report["passed"] = False
    return report
