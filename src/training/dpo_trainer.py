"""
Phase 2 — DPO training with pre-computed reference log-probabilities.
Defaults to precompute_ref_log_probs=True for 24 GB GPU compatibility.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.training.sft_trainer import set_seeds


def precompute_reference_log_probs(
    sft_checkpoint: str,
    preference_dataset: list[dict],
    cache_path: str,
    cfg: dict[str, Any],
) -> str:
    """
    Run the SFT checkpoint once over the preference dataset, cache log-probs.
    Only the policy model is loaded during DPO training — halves VRAM usage.
    """
    import json as _json
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    adapter_cfg_path = Path(sft_checkpoint) / "adapter_config.json"
    if adapter_cfg_path.exists():
        adapter_cfg = _json.loads(adapter_cfg_path.read_text())
        base_name = adapter_cfg["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
        base = AutoModelForCausalLM.from_pretrained(
            base_name, device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, sft_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            sft_checkpoint, device_map="auto", torch_dtype=torch.bfloat16,
        )
    model.eval()

    log_probs = []
    with torch.no_grad():
        for ex in preference_dataset:
            for key in ("chosen", "rejected"):
                text = ex["prompt"] + ex[key]
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                log_probs.append({
                    "prompt": ex["prompt"],
                    "completion": ex[key],
                    "type": key,
                    "log_prob": outputs.loss.item() * -inputs["input_ids"].shape[1],
                })

    torch.save(log_probs, cache_path)
    del model
    torch.cuda.empty_cache()
    return cache_path


def run_dpo(cfg: dict[str, Any], tracker=None) -> str:
    """Train DPO model; returns path to best checkpoint."""
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seeds(seed)

    dpo_cfg = cfg["dpo"]
    t = cfg["training"]
    sft_checkpoint = t["sft_checkpoint"]

    pref_data = _load_preference_dataset(cfg["preference_data"]["preference_cache"])
    pref_ds = Dataset.from_list(pref_data)

    val_n = int(len(pref_data) * cfg["preference_data"].get("val_fraction", 0.1))
    train_ds = pref_ds.select(range(len(pref_ds) - val_n))
    val_ds = pref_ds.select(range(len(pref_ds) - val_n, len(pref_ds)))

    # Load the SFT adapter as the DPO policy starting point — not a fresh LoRA.
    # build_qlora_model initialises lora_B to zero; starting DPO from there means
    # the policy never benefits from SFT training.
    model, tokenizer = _load_sft_as_dpo_policy(sft_checkpoint, cfg)

    dpo_config = DPOConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],  # must be ~10x lower than SFT
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        max_length=t.get("max_seq_length", 2048),
        max_prompt_length=t.get("max_prompt_length", 512),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 100),
        logging_steps=t.get("logging_steps", 10),
        beta=dpo_cfg["beta"],
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        precompute_ref_log_probs=dpo_cfg.get("precompute_ref_log_probs", True),
        seed=seed,
        report_to="none",
    )

    ref_model = None
    if not dpo_cfg.get("precompute_ref_log_probs", True):
        from transformers import AutoModelForCausalLM
        ref_model = AutoModelForCausalLM.from_pretrained(
            sft_checkpoint,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=dpo_config,
    )

    if tracker:
        tracker.start_run(
            run_name=cfg.get("tracking", {}).get("run_name") or "dpo",
            config=cfg,
        )

    trainer.train()

    best_dir = Path(t["output_dir"]) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    if tracker:
        tracker.log(trainer.state.log_history[-1] if trainer.state.log_history else {})
        tracker.finish()

    return str(best_dir)


def _load_sft_as_dpo_policy(sft_checkpoint: str, cfg: dict[str, Any]):
    """Load the SFT adapter onto the quantised base model for use as the DPO policy.

    This preserves the SFT adapter weights (non-zero lora_B) so DPO refines them
    rather than re-learning from a zero-initialised adapter.
    """
    import json as _json
    from pathlib import Path as _Path
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = cfg["model"]
    bnb_cfg = cfg["bnb"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=bnb_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bnb_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, bnb_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
    )

    adapter_cfg_path = _Path(sft_checkpoint) / "adapter_config.json"
    if adapter_cfg_path.exists():
        adapter_cfg = _json.loads(adapter_cfg_path.read_text())
        base_name = adapter_cfg["base_model_name_or_path"]
    else:
        base_name = model_cfg["name"]

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from peft import prepare_model_for_kbit_training
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, sft_checkpoint, is_trainable=True)

    return model, tokenizer


def _load_preference_dataset(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        data = []
        for f in p.glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        data.append(json.loads(line))
        return data
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
