"""
Phase 2 pre-step — Preference dataset generation.

Loads the best SFT checkpoint, generates N completions per prompt at
temperature > 0, scores them via a deterministic rubric (schema validity +
null-case correctness), and writes chosen/rejected JSONL pairs to disk.

Must run before train_dpo.py. Requires a CUDA GPU.

Usage:
    python scripts/generate_preferences.py --config configs/dpo_config.yaml

Override checkpoint or output dir:
    python scripts/generate_preferences.py \\
        --config configs/dpo_config.yaml \\
        --sft-checkpoint ./artifacts/sft/best \\
        --output-dir ./artifacts/dpo/preference_dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data.preference_builder import (
    build_preference_pairs,
    generate_completions_batch,
    sample_prompts,
    save_preference_pairs,
    validate_preference_dataset,
)
from src.manifest.run_manifest import create_manifest


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_schema(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_sft_model(checkpoint: str, cfg: dict):
    """Load SFT LoRA adapter on top of base model for inference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_name = cfg["model"]["name"]
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    print(f"Applying LoRA adapter from: {checkpoint}")
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference dataset from SFT model completions."
    )
    parser.add_argument("--config", required=True, help="Path to dpo_config.yaml")
    parser.add_argument(
        "--sft-checkpoint",
        default=None,
        help="Override SFT checkpoint path from config",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without loading model or generating completions",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Merge base config if defaults key present
    if "defaults" in cfg:
        config_dir = Path(args.config).parent
        base_cfg = {}
        for default in cfg.pop("defaults"):
            base_file = config_dir / f"{default}.yaml"
            if base_file.exists():
                with open(base_file) as bf:
                    base_cfg.update(yaml.safe_load(bf) or {})
        base_cfg.update(cfg)
        cfg = base_cfg

    cfg["_config_path"] = args.config

    pref_cfg = cfg["preference_data"]
    sft_checkpoint = args.sft_checkpoint or cfg["training"]["sft_checkpoint"]
    output_dir = args.output_dir or pref_cfg["preference_cache"]
    output_path = str(Path(output_dir) / "preference_pairs.jsonl")
    val_output_path = str(Path(output_dir) / "preference_pairs_val.jsonl")

    schema_path = cfg["task"]["output_schema_path"]
    schema = _load_schema(schema_path)

    train_examples = _load_jsonl(cfg["dataset"]["train_path"])
    print(f"Loaded {len(train_examples)} training examples")

    sampled = sample_prompts(
        examples=train_examples,
        target_pairs=pref_cfg["target_pairs"],
        null_case_fraction=pref_cfg["null_case_fraction"],
        completions_per_prompt=pref_cfg["completions_per_prompt"],
        seed=cfg.get("reproducibility", {}).get("seed", 42),
    )

    null_count = sum(1 for e in sampled if e.get("is_null_case", False))
    print(
        f"Sampled {len(sampled)} prompts "
        f"({null_count} null-case, {len(sampled) - null_count} positive)"
    )
    print(f"Will generate {pref_cfg['completions_per_prompt']} completions each "
          f"→ {len(sampled) * pref_cfg['completions_per_prompt']} total generations")

    if args.dry_run:
        print("--dry-run: stopping before model load.")
        return

    manifest = create_manifest(
        config=cfg,
        dataset_paths=[cfg["dataset"]["train_path"]],
    )
    print(f"Run ID: {manifest.run_id} | Dataset hash: {manifest.dataset_hash}")

    model, tokenizer = _load_sft_model(sft_checkpoint, cfg)

    prompts = [ex["prompt"] for ex in sampled]
    print("Generating completions...")
    all_completions = generate_completions_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        completions_per_prompt=pref_cfg["completions_per_prompt"],
        max_new_tokens=cfg["training"].get("max_new_tokens", 256),
        temperature=pref_cfg.get("generation_temperature", 0.8),
    )

    pairs = build_preference_pairs(sampled, all_completions, schema)
    print(f"Built {len(pairs)} preference pairs from {len(sampled)} prompts")

    validate_preference_dataset(
        pairs,
        target_pairs=pref_cfg["target_pairs"],
        null_case_fraction=cfg.get("reproducibility", {}).get(
            "tolerances", {}
        ).get("null_accuracy", 0.10),
    )

    # Split into train/val
    val_n = max(1, int(len(pairs) * pref_cfg.get("val_fraction", 0.1)))
    import random
    rng = random.Random(cfg.get("reproducibility", {}).get("seed", 42))
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    train_pairs = shuffled[val_n:]
    val_pairs = shuffled[:val_n]

    save_preference_pairs(train_pairs, output_path)
    save_preference_pairs(val_pairs, val_output_path)

    null_frac = sum(1 for p in pairs if p["is_null_case"]) / len(pairs)
    avg_margin = sum(p["chosen_score"] - p["rejected_score"] for p in pairs) / len(pairs)

    print(f"\nPreference dataset written:")
    print(f"  Train pairs: {len(train_pairs)} → {output_path}")
    print(f"  Val pairs:   {len(val_pairs)} → {val_output_path}")
    print(f"  Null-case fraction: {null_frac:.2%}")
    print(f"  Avg score margin (chosen - rejected): {avg_margin:.2f}")

    # Save generation provenance to manifest
    manifest.metrics.update({
        "sft_checkpoint_used": sft_checkpoint,
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "null_case_fraction": null_frac,
        "avg_score_margin": avg_margin,
        "completions_per_prompt": pref_cfg["completions_per_prompt"],
        "generation_temperature": pref_cfg.get("generation_temperature", 0.8),
        "ranking_strategy": pref_cfg.get("ranking_strategy", "deterministic"),
    })
    manifest_path = Path(output_dir) / "generation_manifest.json"
    manifest.save(manifest_path)
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
