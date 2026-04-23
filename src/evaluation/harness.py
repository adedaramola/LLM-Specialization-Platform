"""
Evaluation harness: pluggable provider, unconstrained + constrained decoding,
metrics.json emission with pass/fail CI gates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import compute_all_metrics


def build_provider(provider_name: str, model_path: str, cfg: dict[str, Any]):
    if provider_name == "hf_native":
        from src.evaluation.providers.hf_provider import HFNativeProvider
        return HFNativeProvider(model_path, cfg.get("hf_native", {}))
    elif provider_name == "vllm":
        from src.evaluation.providers.vllm_provider import VLLMProvider
        return VLLMProvider(model_path, cfg.get("vllm", {}))
    elif provider_name == "ollama":
        from src.evaluation.providers.ollama_provider import OllamaProvider
        model_tag = cfg.get("ollama", {}).get("model_tag") or model_path
        return OllamaProvider(model_tag, cfg.get("ollama", {}))
    elif provider_name == "llama_cpp":
        from src.evaluation.providers.llamacpp_provider import LlamaCppProvider
        return LlamaCppProvider(model_path, cfg.get("llama_cpp", {}))
    elif provider_name == "tgi":
        from src.evaluation.providers.tgi_provider import TGIProvider
        return TGIProvider(model_path, cfg.get("tgi", {}))
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def build_constrained_generator(backend: str, schema: dict):
    if backend == "outlines":
        import outlines
        import outlines.models as om

        def generate_constrained(model, prompts: list[str], max_tokens: int) -> list[str]:
            gen = outlines.generate.json(model, schema)
            return [json.dumps(gen(p)) for p in prompts]

        return generate_constrained
    else:
        raise ValueError(f"Constrained decoding backend '{backend}' not yet implemented.")


def evaluate_model(
    model_label: str,
    model_path: str,
    test_examples: list[dict],
    schema: dict,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    inference_cfg = eval_cfg["inference"]
    provider_name = inference_cfg["provider"]
    gen_cfg = eval_cfg["generation"]

    provider = build_provider(provider_name, model_path, inference_cfg)

    prompts = [ex["prompt"] for ex in test_examples]
    references = [ex["completion"] for ex in test_examples]
    null_labels = [ex.get("is_null_case", False) for ex in test_examples]

    raw_preds = provider.generate(prompts, gen_cfg)
    raw_metrics = compute_all_metrics(raw_preds, references, null_labels, schema)

    constrained_metrics: dict[str, float] = {}
    raw_vs_guided_gap: dict[str, float] = {}

    if eval_cfg.get("constrained", {}).get("enabled", False):
        constrained_preds = _constrained_generate(
            provider, prompts, schema, eval_cfg["constrained"], gen_cfg
        )
        constrained_metrics = compute_all_metrics(
            constrained_preds, references, null_labels, schema
        )
        for k in raw_metrics:
            raw_vs_guided_gap[f"{k}_gap"] = constrained_metrics.get(k, 0) - raw_metrics.get(k, 0)

    return {
        "model": model_label,
        "model_path": model_path,
        "provider": provider_name,
        "raw": raw_metrics,
        "constrained": constrained_metrics,
        "raw_vs_guided_gap": raw_vs_guided_gap,
        "n_examples": len(test_examples),
        "n_positive": sum(1 for x in null_labels if not x),
        "n_null": sum(null_labels),
    }


def _constrained_generate(provider, prompts, schema, constrained_cfg, gen_cfg) -> list[str]:
    backend = constrained_cfg.get("backend", "outlines")
    if backend == "outlines":
        try:
            import outlines
            import outlines.models
            hf_model = getattr(provider, "_model", None)
            hf_tok = getattr(provider, "_tokenizer", None)
            if hf_model is None:
                return [""] * len(prompts)
            om = outlines.models.Transformers(hf_model, hf_tok)
            generator = outlines.Generator(om, outlines.json_schema(schema))
            results = []
            for i, p in enumerate(prompts):
                try:
                    results.append(generator(p))
                except Exception:
                    results.append("")
                if (i + 1) % 40 == 0 or (i + 1) == len(prompts):
                    print(f"  [constrained] {i + 1}/{len(prompts)} examples generated")
            return results
        except Exception as e:
            print(f"  [constrained] skipped: {type(e).__name__}: {e}")
            return [""] * len(prompts)
    return [""] * len(prompts)


def collect_qualitative_samples(
    model_label: str,
    model_path: str,
    test_examples: list[dict],
    schema: dict,
    eval_cfg: dict[str, Any],
    n_samples: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate predictions on a small slice of the test set and categorise
    each example as: easy, hard, null, or failure_mode.

    Categories:
      null         — is_null_case=True examples
      easy         — correct prediction, high schema validity score
      hard         — correct prediction but marginal (schema-valid, field match imperfect)
      failure_mode — incorrect prediction (wrong null/non-null, or schema-invalid)
    """
    import random as _random
    from src.evaluation.metrics import _parse_json_safe

    rng = _random.Random(seed)

    null_ex = [e for e in test_examples if e.get("is_null_case", False)]
    pos_ex  = [e for e in test_examples if not e.get("is_null_case", False)]

    # Aim for ~25% null, 75% positive in the sample
    n_null = min(len(null_ex), max(1, n_samples // 4))
    n_pos  = min(len(pos_ex),  n_samples - n_null)
    sampled = rng.sample(null_ex, n_null) + rng.sample(pos_ex, n_pos)
    rng.shuffle(sampled)

    inference_cfg = eval_cfg["inference"]
    gen_cfg = eval_cfg["generation"]
    provider = build_provider(inference_cfg["provider"], model_path, inference_cfg)
    prompts = [e["prompt"] for e in sampled]
    preds = provider.generate(prompts, gen_cfg)

    records = []
    for ex, pred in zip(sampled, preds):
        is_null = ex.get("is_null_case", False)
        obj, ok = _parse_json_safe(pred)
        pred_null = ok and isinstance(obj, dict) and obj.get("null_extraction", False)
        correct_null = pred_null == is_null

        try:
            import jsonschema as _js
            _js.validate(obj, schema) if ok else (_ for _ in ()).throw(Exception())
            schema_valid = True
        except Exception:
            schema_valid = False

        if is_null:
            category = "null"
        elif not correct_null or not schema_valid:
            category = "failure_mode"
        elif schema_valid and correct_null:
            ref_obj, _ = _parse_json_safe(ex["completion"])
            pred_entities = {e["name"] for e in (obj or {}).get("entities", []) if isinstance(e, dict)}
            ref_entities  = {e["name"] for e in (ref_obj or {}).get("entities", []) if isinstance(e, dict)}
            if pred_entities == ref_entities:
                category = "easy"
            else:
                category = "hard"
        else:
            category = "failure_mode"

        records.append({
            "model": model_label,
            "category": category,
            "is_null_case": is_null,
            "prompt": ex["prompt"],
            "reference": ex["completion"],
            "prediction": pred,
            "schema_valid": schema_valid,
            "correct_null_prediction": correct_null,
        })

    return records


def emit_metrics_json(
    results: list[dict[str, Any]],
    thresholds: dict[str, float],
    output_path: str,
    schema_version: str = "1.0.0",
    gate_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit metrics.json.

    pass_fail — raw decoding metrics vs raw thresholds (research transparency).
    deployment_pass_fail — gate-mode metrics vs deployment thresholds (drives ci_pass).

    The gate mode is set in eval_config.yaml under metrics.ci_gate. It defaults to
    "constrained" so the CI gate matches the production serving mode. Raw pass_fail
    is retained so a reviewer can see unconstrained behaviour without it affecting
    the deployment decision.
    """
    gate_cfg = gate_cfg or {}
    gate_mode = gate_cfg.get("mode", "raw")
    fallback_to_raw = gate_cfg.get("fallback_to_raw", True)
    gate_thresholds = gate_cfg.get("thresholds") or thresholds

    def _check(value: float, key: str, thr: dict) -> bool:
        return value >= thr.get(key, 0.0)

    output: dict[str, Any] = {
        "schema_version": schema_version,
        "ci_gate_mode": gate_mode,
        "models": {},
        "ci_pass": True,
    }

    for result in results:
        label = result["model"]
        raw = result["raw"]
        constrained = result.get("constrained", {})

        # Research transparency: raw pass_fail against raw thresholds
        raw_pass_fail: dict[str, Any] = {}
        for metric_key, value in raw.items():
            if isinstance(value, float):
                raw_pass_fail[metric_key] = {
                    "value": value,
                    "passed": _check(value, metric_key, thresholds),
                }

        # Deployment gate: use gate_mode metrics, fall back to raw if constrained not run
        if gate_mode == "constrained" and constrained:
            gate_metrics = constrained
            effective_mode = "constrained"
        elif fallback_to_raw or gate_mode == "raw":
            gate_metrics = raw
            effective_mode = "raw"
        else:
            gate_metrics = {}
            effective_mode = "none"

        deployment_pass_fail: dict[str, Any] = {}
        for metric_key, value in gate_metrics.items():
            if isinstance(value, float):
                passed = _check(value, metric_key, gate_thresholds)
                deployment_pass_fail[metric_key] = {"value": value, "passed": passed}
                if not passed:
                    output["ci_pass"] = False

        model_entry: dict[str, Any] = {
            "raw": raw,
            "constrained": constrained,
            "raw_vs_guided_gap": result.get("raw_vs_guided_gap", {}),
            "pass_fail": raw_pass_fail,
            "deployment_pass_fail": deployment_pass_fail,
            "deployment_gate_mode": effective_mode,
        }
        output["models"][label] = model_entry

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    return output
