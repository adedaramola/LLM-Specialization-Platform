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
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=cfg.get("device_map", "auto"),
            torch_dtype=torch_dtype,
        )
        self._model.eval()

    def generate(self, prompts: list[str], gen_cfg: dict[str, Any]) -> list[str]:
        import torch
        results = []
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 512),
                    do_sample=gen_cfg.get("do_sample", False),
                    temperature=gen_cfg.get("temperature", 1.0) if gen_cfg.get("do_sample") else 1.0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results
