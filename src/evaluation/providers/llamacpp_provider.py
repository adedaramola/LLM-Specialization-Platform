"""llama-cpp-python inference provider for GGUF artifacts."""
from __future__ import annotations

from typing import Any


class LlamaCppProvider:
    name = "llama_cpp"

    def __init__(self, model_path: str, cfg: dict[str, Any]):
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            n_ctx=cfg.get("n_ctx", 2048),
            n_gpu_layers=cfg.get("n_gpu_layers", -1),  # -1 = all layers on GPU
            verbose=False,
        )

    def generate(self, prompts: list[str], gen_cfg: dict[str, Any]) -> list[str]:
        max_tokens = gen_cfg.get("max_new_tokens", 128)
        temperature = gen_cfg.get("temperature", 0.0)
        batch_size = gen_cfg.get("batch_size", 8)
        log_every = gen_cfg.get("log_every", batch_size * 5)

        results = []
        total = len(prompts)
        for i, prompt in enumerate(prompts):
            out = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            results.append(out["choices"][0]["text"])
            done = i + 1
            if done % log_every == 0 or done == total:
                print(f"  [llama_cpp] {done}/{total} examples generated")
        return results
