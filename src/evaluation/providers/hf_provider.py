"""HuggingFace-native inference provider."""
from __future__ import annotations

from typing import Any


class HFNativeProvider:
    name = "hf_native"

    def __init__(self, model_path: str, cfg: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Left-pad so all sequences in a batch align at the right (generation side)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=cfg.get("device_map", "auto"),
            torch_dtype=torch_dtype,
        )
        self._model.eval()

    def generate(self, prompts: list[str], gen_cfg: dict[str, Any]) -> list[str]:
        import torch

        max_new_tokens = gen_cfg.get("max_new_tokens", 128)
        do_sample = gen_cfg.get("do_sample", False)
        temperature = gen_cfg.get("temperature", 1.0) if do_sample else 1.0
        batch_size = gen_cfg.get("batch_size", 8)
        log_every = gen_cfg.get("log_every", batch_size * 5)

        results: list[str] = []
        total = len(prompts)

        for batch_start in range(0, total, batch_size):
            batch = prompts[batch_start: batch_start + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            for seq in outputs:
                new_tokens = seq[prompt_len:]
                results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))

            done = min(batch_start + batch_size, total)
            if done % log_every == 0 or done == total:
                print(f"  [hf_provider] {done}/{total} examples generated")

        return results
