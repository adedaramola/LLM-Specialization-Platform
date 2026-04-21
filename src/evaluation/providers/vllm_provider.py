"""vLLM inference provider — HTTP API only, no vllm SDK at top level."""
from __future__ import annotations

import json
import urllib.request
from typing import Any


class VLLMProvider:
    name = "vllm"

    def __init__(self, model_path: str, cfg: dict[str, Any]):
        host = cfg.get("host", "localhost")
        port = cfg.get("port", 8000)
        self._url = f"http://{host}:{port}/v1/completions"
        self._model_path = model_path

    def generate(self, prompts: list[str], gen_cfg: dict[str, Any]) -> list[str]:
        results = []
        for prompt in prompts:
            payload = json.dumps({
                "model": self._model_path,
                "prompt": prompt,
                "max_tokens": gen_cfg.get("max_new_tokens", 512),
                "temperature": gen_cfg.get("temperature", 0.0),
                "n": 1,
            }).encode()
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            results.append(data["choices"][0]["text"])
        return results
