"""Text Generation Inference (TGI) provider — HTTP API, no SDK import."""
from __future__ import annotations

import json
import urllib.request
from typing import Any


class TGIProvider:
    name = "tgi"

    def __init__(self, model_path: str, cfg: dict[str, Any]):
        host = cfg.get("host", "localhost")
        port = cfg.get("port", 8080)
        self._url = f"http://{host}:{port}/generate"

    def generate(self, prompts: list[str], gen_cfg: dict[str, Any]) -> list[str]:
        results = []
        for prompt in prompts:
            payload = json.dumps({
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": gen_cfg.get("max_new_tokens", 512),
                    "temperature": gen_cfg.get("temperature", 0.01),
                    "do_sample": gen_cfg.get("do_sample", False),
                },
            }).encode()
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            results.append(data["generated_text"])
        return results
