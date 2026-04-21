"""
Phase 1 — SFT training with QLoRA via TRL's SFTTrainer.
Seeds set for Python, NumPy, PyTorch, CUDA.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


from transformers import TrainerCallback


class GenerationLogCallback(TrainerCallback):
    """
    Logs sample generations at regular step intervals so output drift is
    visible mid-run without waiting for a full evaluation pass.
    Implemented as a plain object that SFTTrainer accepts as a TrainerCallback.
    """

    def __init__(self, tokenizer, val_examples: list[dict], tracker, log_steps: int, n_samples: int):
        self._tokenizer = tokenizer
        self._val_examples = val_examples
        self._tracker = tracker
        self._log_steps = log_steps
        self._n_samples = min(n_samples, len(val_examples))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step == 0:
            return
        if state.global_step % self._log_steps != 0:
            return
        samples = random.sample(self._val_examples, self._n_samples)
        lines = []
        model.eval()
        with torch.no_grad():
            for ex in samples:
                inputs = self._tokenizer(
                    ex["prompt"], return_tensors="pt", truncation=True, max_length=512
                ).to(model.device)
                out = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                pred = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                lines.append(f"PROMPT: {ex['prompt'][:120]!r}\nPRED:   {pred!r}\nREF:    {ex['completion']!r}")
        model.train()
        self._tracker.log_text("generation_samples", "\n\n".join(lines), step=state.global_step)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuBLAS/cuDNN algorithm selection introduces residual non-determinism
    # even with fixed seeds. Results are reproducible within tolerance, not bit-exact.
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_qlora_model(model_cfg: dict, lora_cfg: dict, bnb_cfg: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=bnb_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bnb_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, bnb_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],       # fixed across all rank experiments
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        use_rslora=lora_cfg.get("use_rslora", False),
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def run_sft(cfg: dict[str, Any], tracker=None) -> str:
    """Train SFT model; returns path to best checkpoint."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    seed = cfg.get("reproducibility", {}).get("seed", 42)
    set_seeds(seed)

    model, tokenizer = build_qlora_model(cfg["model"], cfg["lora"], cfg["bnb"])

    train_data = _load_split(cfg["dataset"]["train_path"])
    val_data = _load_split(cfg["dataset"]["val_path"])
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    t = cfg["training"]
    sft_config = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t.get("weight_decay", 0.0),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        max_seq_length=t.get("max_seq_length", 2048),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 200),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 200),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        logging_steps=t.get("logging_steps", 10),
        seed=seed,
        packing=False,
        report_to="none",
    )

    def formatting_fn(examples):
        return [p + c for p, c in zip(examples["prompt"], examples["completion"])]

    callbacks = []
    log_steps = t.get("generation_log_steps", 0)
    log_samples = t.get("generation_log_samples", 4)
    if log_steps > 0 and tracker:
        callbacks.append(
            GenerationLogCallback(
                tokenizer=tokenizer,
                val_examples=val_data,
                tracker=tracker,
                log_steps=log_steps,
                n_samples=log_samples,
            )
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        formatting_func=formatting_fn,
        callbacks=callbacks or None,
    )

    if tracker:
        tracker.start_run(
            run_name=cfg.get("tracking", {}).get("run_name") or "sft",
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


def _load_split(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
