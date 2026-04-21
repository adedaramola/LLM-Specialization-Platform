"""Ollama inference provider — HTTP API, no SDK import."""
from __future__ import annotations

import json
import urllib.request
from typing import Any


class OllamaProvider:
    name = "ollama"

    def __init__(self, model_tag: str, cfg: dict[str, Any]):
        host = cfg.get("host", "localhost")
        port = cfg.get("port", 11434)
        self._url = f"http://{host}:{port}/api/generate"
        self._model_tag = model_tag

    def generate(self, prompts: list[str], gen_cfg: dict[str, Any]) -> list[str]:
        results = []
        for prompt in prompts:
            payload = json.dumps({
                "model": self._model_tag,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": gen_cfg.get("max_new_tokens", 512),
                    "temperature": gen_cfg.get("temperature", 0.0),
                },
            }).encode()
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            results.append(data["response"])
        return results
