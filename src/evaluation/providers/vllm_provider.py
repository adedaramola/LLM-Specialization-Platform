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
        self._completions_url = f"http://{host}:{port}/v1/completions"
        # The server registers its model under the exact string it was launched
        # with (often an absolute path); a config-relative path 404s. Ask the
        # server what it serves rather than assuming.
        self._model_path = self._served_model_id(host, port) or model_path

    @staticmethod
    def _served_model_id(host: str, port: int) -> str | None:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=10) as resp:
                data = json.loads(resp.read())
            models = data.get("data", [])
            return models[0]["id"] if models else None
        except Exception:
            return None

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._completions_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    def _batch_complete(
        self,
        prompts: list[str],
        gen_cfg: dict[str, Any],
        extra_params: dict | None = None,
    ) -> list[str]:
        """Send prompts in chunks and collect results in order."""
        batch_size = gen_cfg.get("batch_size", 32)
        max_tokens = gen_cfg.get("max_new_tokens", 512)
        temperature = gen_cfg.get("temperature", 0.0)

        results: list[str] = []
        total = len(prompts)

        for start in range(0, total, batch_size):
            chunk = prompts[start : start + batch_size]
            payload: dict[str, Any] = {
                "model": self._model_path,
                "prompt": chunk,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": 1,
            }
            if extra_params:
                payload.update(extra_params)

            response = self._post(payload)
            # choices are returned in prompt order; sort by index to be safe
            choices = sorted(response["choices"], key=lambda c: c["index"])
            results.extend(c["text"] for c in choices)

            done = min(start + batch_size, total)
            print(f"  [vllm] {done}/{total} examples generated")

        return results

    def generate(self, prompts: list[str], gen_cfg: dict[str, Any]) -> list[str]:
        return self._batch_complete(prompts, gen_cfg)

    def generate_constrained(
        self, prompts: list[str], schema: dict, gen_cfg: dict[str, Any]
    ) -> list[str]:
        """Constrained decoding via vLLM's native guided_json — batched."""
        return self._batch_complete(
            prompts, gen_cfg, extra_params={"guided_json": json.dumps(schema)}
        )
