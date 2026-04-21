"""
Dataset preparation: load raw data, generate negative examples,
split 80/10/10, decontaminate, content-hash, serialize to JSONL.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.data.decontamination import decontaminate
from src.data.storage import StorageBackend, build_storage


CHAT_TEMPLATE_EXTRACTION = (
    "<|im_start|>system\n"
    "You are a precise JSON extractor. Extract structured data according to the schema. "
    "If no extractable data is present, respond with {{\"null_extraction\": true}}.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n{input}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def format_example(example: dict[str, Any], task: str = "json_extraction") -> dict[str, Any]:
    prompt = CHAT_TEMPLATE_EXTRACTION.format(input=example["input"])
    target = json.dumps(example["output"], ensure_ascii=False)
    return {
        "prompt": prompt,
        "completion": target,
        "is_null_case": example.get("output", {}).get("null_extraction", False),
        "metadata": example.get("metadata", {}),
    }


def split_dataset(
    examples: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def validate_null_fraction(
    examples: list[dict], min_fraction: float = 0.10
) -> None:
    null_count = sum(1 for ex in examples if ex.get("is_null_case", False))
    fraction = null_count / len(examples) if examples else 0.0
    if fraction < min_fraction:
        raise ValueError(
            f"Null-case fraction {fraction:.2%} is below minimum {min_fraction:.2%}. "
            "Add more negative examples."
        )


def content_hash_dataset(examples: list[dict]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(json.dumps(ex, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def build_and_save_dataset(
    raw_examples: list[dict[str, Any]],
    cfg: dict[str, Any],
    storage: StorageBackend,
) -> dict[str, Any]:
    dataset_cfg = cfg["dataset"]
    null_min = dataset_cfg.get("null_case_fraction", 0.10)
    seed = dataset_cfg.get("seed", 42)
    ratios = tuple(dataset_cfg.get("split_ratios", [0.8, 0.1, 0.1]))

    formatted = [format_example(ex, cfg["task"]["name"]) for ex in raw_examples]

    train_raw, val_raw, test_raw = split_dataset(formatted, ratios, seed)

    train_clean, val_clean, decontam_stats = decontaminate(train_raw, val_raw, test_raw)

    for split_name, split in [("train", train_clean), ("val", val_clean), ("test", test_raw)]:
        validate_null_fraction(split, null_min)

    train_hash = content_hash_dataset(train_clean)
    val_hash = content_hash_dataset(val_clean)
    test_hash = content_hash_dataset(test_raw)

    for path_key, data in [
        ("train_path", train_clean),
        ("val_path", val_clean),
        ("test_path", test_raw),
    ]:
        out_path = dataset_cfg[path_key]
        lines = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in data)
        storage.write_text(out_path, lines)

    return {
        "train_size": len(train_clean),
        "val_size": len(val_clean),
        "test_size": len(test_raw),
        "train_hash": train_hash,
        "val_hash": val_hash,
        "test_hash": test_hash,
        "combined_hash": content_hash_dataset(train_clean + val_clean + test_raw),
        "decontamination_stats": decontam_stats,
    }


def load_jsonl(path: str, storage: StorageBackend) -> list[dict]:
    text = storage.read_text(path)
    return [json.loads(line) for line in text.strip().splitlines() if line.strip()]
