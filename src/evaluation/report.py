"""
Fills the model card template (templates/model_card.md) with actual metrics
from metrics.json, the run manifest, and the pipeline config.

All {{placeholder}} tokens in the template are replaced; any that remain
unfilled are left as-is so the omission is visible rather than silently empty.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, decimals: int = 3) -> str:
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.001:
            return f"{value:.2e}"
        return f"{value:.{decimals}f}"
    if value is None:
        return "—"
    return str(value)


def _metric(metrics_data: dict, model: str, key: str, mode: str = "raw") -> str:
    entry = metrics_data.get("models", {}).get(model, {})
    val = entry.get(mode, {}).get(key)
    if val is None:
        val = entry.get("pass_fail", {}).get(key, {}).get("value")
    return _fmt(val)


def _regression_metric(metrics_data: dict, role: str, key: str) -> str:
    """Read from metrics_data["regression"][role][key] (standalone regression block)."""
    val = metrics_data.get("regression", {}).get(role, {}).get(key)
    return _fmt(val)


def _regression_delta(metrics_data: dict, benchmark: str) -> str:
    """Read pre-computed delta from metrics_data["regression"]["deltas"][benchmark]["delta"]."""
    val = metrics_data.get("regression", {}).get("deltas", {}).get(benchmark, {}).get("delta")
    return _fmt(val)


def _delta(metrics_data: dict, model_a: str, model_b: str, key: str) -> str:
    """Compute model_b[key] - model_a[key] in raw metrics."""
    def _get(m):
        return metrics_data.get("models", {}).get(m, {}).get("raw", {}).get(key)
    a, b = _get(model_a), _get(model_b)
    if a is None or b is None:
        return "—"
    return _fmt(b - a, decimals=3)


def _parse_driver(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return raw.strip()


def _parse_cuda_toolkit(raw: str) -> str:
    import re
    m = re.search(r"release (\d+\.\d+)", raw)
    return m.group(1) if m else raw.splitlines()[0].strip()


def _resolve_split_size(manifests: list[dict], ds: dict, metric_key: str, path_key: str) -> Any:
    """Resolve train/val/test size: manifests → ds field → line count → None."""
    for manifest in manifests:
        val = manifest.get("metrics", {}).get(metric_key)
        if val is not None:
            return val
    from_cfg = ds.get(metric_key)
    if from_cfg is not None:
        return from_cfg
    path = ds.get(path_key)
    if path and Path(path).is_file():
        with open(path) as f:
            return sum(1 for _ in f)
    return None


def _load_manifest(path: str | None) -> dict[str, Any]:
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}


def generate_report(
    template_path: str,
    metrics_json_path: str,
    manifest_path: str | None,
    config: dict[str, Any],
    output_path: str,
    sft_config_path: str | None = None,
    sft_manifest_path: str | None = None,
    pref_manifest_path_override: str | None = None,
) -> str:
    """
    Fill the model card template and write the report to output_path.
    Returns the rendered markdown string.
    """
    template = Path(template_path).read_text(encoding="utf-8")

    with open(metrics_json_path) as f:
        metrics_data = json.load(f)

    # Load whichever manifests exist; DPO manifest takes priority for shared fields
    dpo_manifest = _load_manifest(manifest_path)
    sft_manifest = _load_manifest(sft_manifest_path)
    # For hardware/git/hashes: prefer DPO manifest, fall back to SFT manifest
    manifest = dpo_manifest if dpo_manifest else sft_manifest

    # SFT config: explicit sft_config_path, or fall back to SFT manifest's embedded config
    sft_cfg: dict[str, Any] = {}
    if sft_config_path and Path(sft_config_path).exists():
        import yaml
        with open(sft_config_path) as f:
            sft_cfg = yaml.safe_load(f) or {}
    elif sft_manifest:
        sft_cfg = {"training": sft_manifest.get("config", {}).get("training", {})}

    # Preference generation manifest for DPO pair counts and DPO training hyperparams
    _pref_manifest_path = Path(pref_manifest_path_override) if pref_manifest_path_override else (
        Path(config.get("preference_data", {}).get(
            "preference_cache", "./artifacts/dpo/preference_dataset"
        )) / "generation_manifest.json"
    )
    pref_manifest: dict[str, Any] = {}
    if _pref_manifest_path.exists():
        with open(_pref_manifest_path) as f:
            pref_manifest = json.load(f)

    hw = manifest.get("hardware", {})
    lic = manifest.get("licensing", {})
    ds = config.get("dataset", {})
    lora = config.get("lora", {})
    # training section: current config (DPO eval config has no training) → pref_manifest config → {}
    t = config.get("training", {}) or pref_manifest.get("config", {}).get("training", {})
    # SFT training section: explicit SFT config preferred
    sft_t = sft_cfg.get("training") or t
    dpo = config.get("dpo", {}) or pref_manifest.get("config", {}).get("dpo", {})
    pref = config.get("preference_data", {}) or pref_manifest.get("config", {}).get("preference_data", {})
    repro = config.get("reproducibility", {})

    # Tokenizer audit (written by scripts/tokenizer_audit.py)
    tok_audit: dict[str, Any] = {}
    _tok_path = config.get("tokenizer_audit_path", "./artifacts/tokenizer_audit.json")
    if _tok_path and Path(_tok_path).is_file():
        with open(_tok_path) as f:
            tok_audit = json.load(f)

    # Licensing: fall back from manifest → base_config fields
    _license = lic.get("license") or config.get("model", {}).get("license", "")
    _commercial = "Permitted" if "apache" in _license.lower() else "See license"

    # Dataset sources/licenses: show a useful default when lists are empty (synthetic data)
    _sources = ", ".join(ds.get("sources", [])) or "Synthetically generated"
    _ds_licenses = ", ".join(ds.get("licenses", [])) or (
        "N/A (synthetic)" if not ds.get("sources") else "—"
    )

    manifests_list = [m for m in [dpo_manifest, sft_manifest] if m]

    replacements: dict[str, str] = {
        # Header
        "model_name": config.get("model", {}).get("name", "—").split("/")[-1] + "-specialized",
        "base_model": config.get("model", {}).get("name", "—"),
        "task": config.get("task", {}).get("description", config.get("task", {}).get("name", "—")),
        "pipeline_version": config.get("pipeline_version", "—"),
        "run_id": manifest.get("run_id", "—"),
        "git_commit": (manifest.get("git_commit") or "—")[:12],  # short SHA
        # Licensing
        "base_model_license": _license or "—",
        "commercial_use": _commercial,
        "attribution": lic.get("attribution") or config.get("model", {}).get("attribution", "—"),
        "restrictions": (
            lic.get("restrictions") or config.get("model", {}).get("license_restrictions", "None") or "None"
        ),
        "dataset_licenses": _ds_licenses,
        # Dataset
        "dataset_sources": _sources,
        "train_size": _fmt(_resolve_split_size(manifests_list, ds, "train_size", "train_path")),
        "val_size": _fmt(_resolve_split_size(manifests_list, ds, "val_size", "val_path")),
        "test_size": _fmt(_resolve_split_size(manifests_list, ds, "test_size", "test_path")),
        "null_fraction": _fmt(ds.get("null_case_fraction", 0.15)),
        "dataset_hash": manifest.get("dataset_hash", "—"),
        "synthetic_generation_model": ds.get("synthetic_generation_model") or "None",
        "ngram_n": str(ds.get("decontamination", {}).get("ngram_n", "—")),
        # Tokenizer — from tokenizer audit manifest; falls back to "—" if audit not run
        "tokenizer_class": tok_audit.get("tokenizer_class", "—"),
        "vocab_size": str(tok_audit["vocab_size"]) if "vocab_size" in tok_audit else "—",
        "chat_template_present": (
            "Yes" if tok_audit.get("chat_template_present") else
            "No" if "chat_template_present" in tok_audit else "—"
        ),
        "added_tokens": (
            (", ".join(tok_audit["added_tokens"]) or "None")
            if "added_tokens" in tok_audit else "—"
        ),
        "byte_fallback_chars": (
            (", ".join(tok_audit["byte_fallback_chars"]) or "None")
            if "byte_fallback_chars" in tok_audit else "—"
        ),
        # SFT hyperparameters — prefer explicit SFT config, fall back to manifest config
        "lora_rank": _fmt(lora.get("rank") or sft_manifest.get("config", {}).get("lora", {}).get("rank", 32)),
        "lora_alpha": _fmt(lora.get("alpha") or sft_manifest.get("config", {}).get("lora", {}).get("alpha", 32)),
        "lora_dropout": _fmt(lora.get("dropout", 0.05)),
        "use_rslora": str(lora.get("use_rslora", False)),
        "target_modules": ", ".join(
            lora.get("target_modules") or
            sft_manifest.get("config", {}).get("lora", {}).get("target_modules", [])
        ),
        "sft_lr": _fmt(sft_t.get("learning_rate")),
        "lr_scheduler": sft_t.get("lr_scheduler_type", "cosine"),
        "sft_epochs": _fmt(sft_t.get("num_train_epochs")),
        "effective_batch": _fmt(
            (sft_t.get("per_device_train_batch_size") or 4) *
            (sft_t.get("gradient_accumulation_steps") or 8)
        ),
        "precision": "bf16" if sft_t.get("bf16", True) else "fp16",
        "gradient_checkpointing": str(sft_t.get("gradient_checkpointing", True)),
        # DPO hyperparameters — prefer pref_manifest config when main config missing DPO details
        "dpo_beta": _fmt(dpo.get("beta", 0.1)),
        "dpo_lr": _fmt(t.get("learning_rate")),
        "dpo_epochs": _fmt(t.get("num_train_epochs", 1)),
        "num_preference_pairs": _fmt(pref_manifest.get("metrics", {}).get("total_pairs", pref.get("target_pairs"))),
        "null_pair_fraction": _fmt(pref.get("null_case_fraction", 0.20)),
        "precompute_ref_log_probs": str(dpo.get("precompute_ref_log_probs", True)),
        "ranking_strategy": pref.get("ranking_strategy", "deterministic"),
        # Per-artifact metrics
        "base_schema_validity": _metric(metrics_data, "base", "schema_validity"),
        "base_field_f1": _metric(metrics_data, "base", "field_f1"),
        "base_exact_match": _metric(metrics_data, "base", "exact_match"),
        "base_null_accuracy": _metric(metrics_data, "base", "null_accuracy"),
        "sft_schema_validity": _metric(metrics_data, "sft", "schema_validity"),
        "sft_field_f1": _metric(metrics_data, "sft", "field_f1"),
        "sft_exact_match": _metric(metrics_data, "sft", "exact_match"),
        "sft_null_accuracy": _metric(metrics_data, "sft", "null_accuracy"),
        "sft_constrained_schema_validity": _metric(metrics_data, "sft", "schema_validity", "constrained"),
        "sft_constrained_field_f1": _metric(metrics_data, "sft", "field_f1", "constrained"),
        "dpo_schema_validity": _metric(metrics_data, "dpo", "schema_validity"),
        "dpo_field_f1": _metric(metrics_data, "dpo", "field_f1"),
        "dpo_exact_match": _metric(metrics_data, "dpo", "exact_match"),
        "dpo_null_accuracy": _metric(metrics_data, "dpo", "null_accuracy"),
        "dpo_constrained_schema_validity": _metric(metrics_data, "dpo", "schema_validity", "constrained"),
        "dpo_constrained_field_f1": _metric(metrics_data, "dpo", "field_f1", "constrained"),
        "merged_schema_validity": _metric(metrics_data, "merged_bf16", "schema_validity"),
        "merged_field_f1": _metric(metrics_data, "merged_bf16", "field_f1"),
        "merged_exact_match": _metric(metrics_data, "merged_bf16", "exact_match"),
        "merged_null_accuracy": _metric(metrics_data, "merged_bf16", "null_accuracy"),
        "gguf_q8_schema_validity": _metric(metrics_data, "gguf_q8", "schema_validity"),
        "gguf_q8_field_f1": _metric(metrics_data, "gguf_q8", "field_f1"),
        "gguf_q8_exact_match": _metric(metrics_data, "gguf_q8", "exact_match"),
        "gguf_q8_null_accuracy": _metric(metrics_data, "gguf_q8", "null_accuracy"),
        "gguf_q4_schema_validity": _metric(metrics_data, "gguf_q4", "schema_validity"),
        "gguf_q4_field_f1": _metric(metrics_data, "gguf_q4", "field_f1"),
        "gguf_q4_exact_match": _metric(metrics_data, "gguf_q4", "exact_match"),
        "gguf_q4_null_accuracy": _metric(metrics_data, "gguf_q4", "null_accuracy"),
        # Degradation deltas
        "adapter_to_merged_f1_delta": _delta(metrics_data, "dpo", "merged_bf16", "field_f1"),
        "adapter_to_merged_null_delta": _delta(metrics_data, "dpo", "merged_bf16", "null_accuracy"),
        "merged_to_q8_f1_delta": _delta(metrics_data, "merged_bf16", "gguf_q8", "field_f1"),
        "merged_to_q8_null_delta": _delta(metrics_data, "merged_bf16", "gguf_q8", "null_accuracy"),
        "q8_to_q4_f1_delta": _delta(metrics_data, "gguf_q8", "gguf_q4", "field_f1"),
        "q8_to_q4_null_delta": _delta(metrics_data, "gguf_q8", "gguf_q4", "null_accuracy"),
        # Regression — stored at metrics_data["regression"], not metrics_data["models"]
        "base_mmlu": _regression_metric(metrics_data, "base", "mmlu_accuracy"),
        "sft_mmlu": _regression_metric(metrics_data, "sft", "mmlu_accuracy"),
        "dpo_mmlu": _regression_metric(metrics_data, "dpo", "mmlu_accuracy"),
        "dpo_mmlu_delta": _regression_delta(metrics_data, "mmlu"),
        "base_hellaswag": _regression_metric(metrics_data, "base", "hellaswag_accuracy"),
        "sft_hellaswag": _regression_metric(metrics_data, "sft", "hellaswag_accuracy"),
        "dpo_hellaswag": _regression_metric(metrics_data, "dpo", "hellaswag_accuracy"),
        "dpo_hellaswag_delta": _regression_delta(metrics_data, "hellaswag"),
        # Reproducibility
        "config_path": config.get("_config_path", "configs/sft_config.yaml"),
        "lockfile_hash": manifest.get("lockfile_hash", "—"),
        "seed": _fmt(repro.get("seed", 42)),
        "f1_tolerance": _fmt(repro.get("tolerances", {}).get("f1", 0.015)),
        "null_tolerance": _fmt(repro.get("tolerances", {}).get("null_accuracy", 0.020)),
        "gpu_model": hw.get("gpu_model", "—"),
        "gpu_count": _fmt(hw.get("gpu_count")),
        "nvidia_driver": _parse_driver(hw.get("nvidia_driver", "—")),
        "cuda_toolkit": _parse_cuda_toolkit(hw.get("cuda_toolkit", "—")),
        "pytorch_cuda": hw.get("pytorch_cuda", "—"),
        "cudnn_version": hw.get("cudnn_version", "—"),
        # Inference examples
        "hf_repo": lic.get("base_model") or config.get("model", {}).get("name", "your-hf-repo"),
        "ollama_model_tag": config.get("model", {}).get("name", "—").split("/")[-1].lower(),
        "intended_use": config.get("task", {}).get("description", "Structured JSON extraction from text."),
    }

    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(rendered, encoding="utf-8")
    return rendered
