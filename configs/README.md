# configs/

Pipeline configuration for all training and evaluation phases.

## Inheritance model

Every stage config declares a `defaults` key that is merged at load time:

```
base_config.yaml          ← shared defaults for all stages
    ↑
sft_config.yaml           ← SFT overrides only
dpo_config.yaml           ← DPO overrides only
eval_config.yaml          ← evaluation overrides only
```

The override config wins on every top-level key. Loading is handled by
`src/config.py:load_config()` — all pipeline scripts call this; none
re-implement the merge logic themselves.

## Files

### `base_config.yaml`
Shared defaults: model name and license metadata, dataset paths and split
ratios, storage backend, experiment tracking, and reproducibility tolerances.
Fill `dataset.sources`, `dataset.licenses`, and `dataset.synthetic_generation_model`
before training — these propagate to the model card and run manifest.

### `sft_config.yaml`
Phase 1 overrides: LoRA rank/alpha/dropout, QLoRA quantization (NF4, double
quant), training hyperparameters. The SFT config is the root config for
`make train` — DPO and eval configs inherit base via their own `defaults` key,
not from SFT.

Key constraints to preserve across experiments:
- `lora.alpha` must stay fixed (32) when sweeping rank — alpha/rank is the
  effective scaling factor and must be a deliberate variable, not incidental
- `lora.alpha` in `dpo_config.yaml` must match SFT exactly

### `dpo_config.yaml`
Phase 2 overrides: DPO beta, learning rate (an order of magnitude below SFT —
the most common misconfiguration), preference dataset construction settings.

`precompute_ref_log_probs: true` is the default and required for single-GPU
setups — standard DPO loads two model copies and OOMs a 24 GB card for 7B+
models. Set to false only on A100/H100 with sufficient VRAM.

### `eval_config.yaml`
Evaluation provider, generation settings, per-artifact paths, and CI gate
thresholds. Two threshold sections:

- `metrics.thresholds` — raw decoding thresholds, used for `pass_fail` in
  `metrics.json` (research transparency)
- `metrics.ci_gate.thresholds` — constrained decoding thresholds, used for
  `deployment_pass_fail` and drives `ci_pass` (deployment gate)

Change `inference.provider` to switch between `hf_native`, `vllm`, `ollama`,
`tgi`, or `llama_cpp` without any code changes. `llama_cpp` is selected
automatically for GGUF artifact evaluation.

### `schemas/extraction_schema.json`
JSON Schema (Draft-07) for the extraction task output. Enforces:
- `null_extraction` (boolean) is always required
- When `null_extraction: false`, `entities` is required and must have ≥ 1 item
- Each entity requires `name`, `type`, `value`; `confidence` is optional

This schema is used by three components: the evaluation harness (schema
validity metric), the preference builder (deterministic scoring rubric), and
constrained decoding (outlines/xgrammar guarantees outputs conform to it).

## Adapting to a new task

1. Replace `schemas/extraction_schema.json` with your task's output schema
2. Update `base_config.yaml` → `task.description` and `task.output_schema_path`
3. Update `dataset.sources`, `dataset.licenses`, `dataset.synthetic_generation_model`
4. Adjust `eval_config.yaml` → `metrics.thresholds` and `metrics.ci_gate.thresholds`
   for your task's expected performance range

No script changes required.
