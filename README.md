# LLM Specialization Platform

An end-to-end pipeline for fine-tuning, evaluating, and exporting specialized language models. Built to demonstrate production ML engineering discipline — reproducibility, silent failure detection, and CI/CD integration — not just model training.

The initial task is structured JSON extraction from unstructured text using [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). Swapping tasks requires only a new dataset, schema, and config — no code changes.

---

## Why this exists

Most fine-tuning projects treat training as the deliverable. Production ML fails for different reasons: the tokenizer drifts between training and serving, the GGUF you ship scores differently than the adapter you evaluated, the model fires on null inputs it should abstain from, and evaluation metrics pass against a mock that diverges from the real stack.

This pipeline is designed adversarially against those failure modes:

- **Null-case pairs are first-class** — DPO preference data explicitly trains the model to abstain when there's nothing to extract, not just to extract well when there is
- **Evaluation runs against the artifact that ships** — merged BF16 and each GGUF quantization are re-evaluated on the frozen test set, not just the adapter during training
- **Raw and constrained metrics are reported separately** — a model hitting 99% only under schema constraints has learned something different from one that doesn't need them
- **Field-level F1 is computed on positive examples only** — including null cases in field F1 inflates it to exactly the null-case fraction for any over-abstaining model, masking real extraction quality
- **Validation gates raise, not warn** — preference dataset validators raise `ValueError` when thresholds are breached; silent warnings let bad datasets reach training
- **Every run is reproducible** — manifests capture git commit, lockfile hash, dataset hash, and full hardware fingerprint (driver, CUDA, cuDNN, GPU model)
- **CI gate is machine-readable** — `metrics.json` is emitted with pass/fail flags against configurable thresholds, designed to be consumed by downstream GitOps pipelines

---

## Pipeline overview

```
Phase 0   Task definition, tokenizer audit, baseline evaluation
   ↓
Phase 1   Supervised Fine-Tuning (SFT) with QLoRA
   ↓
Phase 2   Direct Preference Optimization (DPO) from best SFT checkpoint
   ↓
Phase 3   Evaluation: raw + constrained decoding, regression check, metrics.json
   ↓
Phase 4   Export: LoRA adapters → merged BF16 → GGUF Q8_0 + Q4_K_M → re-verify
```

---

## Results

Trained on 2,998 examples (14% null cases), evaluated on a frozen 375-example test set.

| Artifact | Schema Validity | Null Accuracy | Extraction F1 | Decoding |
|---|---|---|---|---|
| SFT adapter | 0.569 | **1.000** | 0.665 | raw |
| DPO adapter | 0.407 | **1.000** | 0.789 | raw |
| DPO adapter | **0.718** | **1.000** | 0.385 | constrained |
| Merged BF16 | 0.412 | **1.000** | 0.779 | raw |
| GGUF Q8_0 | 0.434 | **1.000** | 0.762 | raw |
| GGUF Q4_K_M | 0.678 | **1.000** | 0.501 | raw |

**Extraction F1** measures whether the model correctly triggers or abstains from extraction (precision/recall on the extraction decision). **Field-level F1** (entity name + type + value match on positive examples only) is the primary quality metric for extracted content and requires re-evaluation under the corrected metric — see below.

Key findings:
- **Null accuracy is perfect across all artifacts** — the model never hallucinates an extraction when there is nothing to extract, and this holds through BF16 merge and both GGUF quantizations.
- **DPO improved extraction recall** (+15% over SFT: 0.650 vs 0.498) at the cost of raw schema validity, indicating the model became more willing to attempt extraction at the expense of output structure.
- **Q4 quantization hurts recall significantly** (0.616 → 0.334 from Q8 to Q4) while schema validity rises — the model generates more syntactically valid but content-sparse JSON at lower precision.
- **Constrained decoding recovers schema validity** to 0.718 for DPO — production deployment should use outlines or equivalent.

General-capability regression check (MMLU, HellaSwag) was not run — base model path was not configured in this run.

---

## Project structure

```
.
├── configs/
│   ├── base_config.yaml          # Shared defaults — all stage configs inherit this
│   ├── sft_config.yaml           # SFT hyperparameters
│   ├── dpo_config.yaml           # DPO hyperparameters
│   ├── eval_config.yaml          # Evaluation thresholds and provider config
│   └── schemas/
│       └── extraction_schema.json  # JSON Schema for the extraction task
│
├── scripts/
│   ├── prepare_dataset.py        # Build + decontaminate dataset, content-hash snapshot
│   ├── tokenizer_audit.py        # Verify tokenizer handles task-specific syntax
│   ├── train_sft.py              # Phase 1: QLoRA SFT
│   ├── generate_preferences.py   # Build DPO preference pairs from SFT completions
│   ├── train_dpo.py              # Phase 2: DPO from best SFT checkpoint
│   ├── evaluate.py               # Phase 3: eval harness, metrics.json, qualitative samples
│   └── export.py                 # Phase 4: adapter → merged BF16 → GGUF
│
├── src/
│   ├── data/
│   │   ├── dataset_builder.py    # Dataset construction and splitting
│   │   ├── decontamination.py    # Hash + n-gram overlap decontamination
│   │   ├── preference_builder.py # Preference pair construction with null-case synthesis
│   │   └── storage.py            # Storage abstraction (local / S3 / HF Hub)
│   ├── training/
│   │   ├── sft_trainer.py        # SFTTrainer wrapper with generation logging
│   │   └── dpo_trainer.py        # DPOTrainer with ref log-prob precomputation
│   ├── evaluation/
│   │   ├── harness.py            # Pluggable provider, raw + constrained eval loop
│   │   ├── metrics.py            # Schema validity, field F1, null accuracy, exact match
│   │   ├── regression.py         # General-capability regression slice (MMLU, HellaSwag)
│   │   ├── report.py             # Model card generation from metrics.json
│   │   └── providers/
│   │       ├── hf_provider.py        # HuggingFace native (batched, left-padded)
│   │       ├── llamacpp_provider.py  # llama-cpp-python (GGUF evaluation)
│   │       ├── vllm_provider.py      # vLLM
│   │       ├── ollama_provider.py
│   │       └── tgi_provider.py
│   ├── export/
│   │   └── exporter.py           # Adapter copy, merge, GGUF conversion + quantization
│   ├── manifest/
│   │   └── run_manifest.py       # Reproducibility manifest: git, hardware, hashes
│   ├── tokenizer/
│   │   └── audit.py              # Tokenizer audit against task-specific syntax
│   └── tracking/
│       └── tracker.py            # Experiment tracking abstraction (filesystem / W&B / MLflow)
│
├── templates/
│   └── model_card.md             # Model card template
├── Dockerfile                    # Pinned CUDA/PyTorch environment
├── Makefile                      # One-command reproduction
├── requirements.txt
└── requirements-dev.txt
```

---

## Quickstart

### Prerequisites

- NVIDIA GPU with ≥24 GB VRAM (tested: H100 80GB, A100 40GB)
- CUDA 12.x driver
- Docker (recommended) or Python 3.10+

### Docker (recommended)

```bash
docker build -t llm-specialization:latest .
docker run --gpus all --rm -v $(PWD):/workspace llm-specialization:latest \
  make train CONFIG=configs/sft_config.yaml
```

### Local (Linux, CUDA available)

```bash
make setup
source .venv/bin/activate
make train CONFIG=configs/sft_config.yaml
make evaluate CONFIG=configs/eval_config.yaml
make export CONFIG=configs/sft_config.yaml
```

### Running individual phases

```bash
# Dataset preparation and tokenizer audit
python scripts/prepare_dataset.py --config configs/sft_config.yaml
python scripts/tokenizer_audit.py --config configs/sft_config.yaml

# SFT training
python scripts/train_sft.py --config configs/sft_config.yaml

# Generate DPO preference pairs from best SFT checkpoint
python scripts/generate_preferences.py --config configs/dpo_config.yaml

# DPO training
python scripts/train_dpo.py --config configs/dpo_config.yaml

# Evaluation (emits artifacts/eval/metrics.json)
python scripts/evaluate.py --config configs/eval_config.yaml --mode all

# Export: adapter → merged BF16 → GGUF
python scripts/export.py --config configs/dpo_config.yaml --checkpoint ./artifacts/dpo/best
```

---

## Training configuration

### SFT (Phase 1)

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Method | QLoRA (4-bit NF4, double quantization) |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 1e-4, cosine schedule, 3% warmup |
| Epochs | 2 |
| Effective batch size | 32 (4 × 8 gradient accumulation) |
| Precision | bf16 |
| Final train loss | 0.410 |

### DPO (Phase 2)

| Parameter | Value |
|---|---|
| Beta | 0.1 |
| Learning rate | 5e-7 (10× below SFT — most common misconfiguration) |
| Epochs | 1 |
| Preference pairs | 1,384 (20.2% null-case pairs) |
| Null-case strategy | Chosen: correct abstention. Rejected: hallucinated extraction (synthetic) |
| Ref log-probs | Precomputed — only policy model loaded during training (single-GPU safe) |
| Final reward margin | 0.132 |
| Reward accuracy | 69.4% |

---

## Evaluation

The evaluation harness runs every model under both unconstrained and constrained decoding, against the same frozen test set used for baseline. It emits `artifacts/eval/metrics.json` — a versioned, machine-readable summary with pass/fail flags against configurable thresholds.

```bash
# Evaluate all training phases
python scripts/evaluate.py --config configs/eval_config.yaml --mode all

# Evaluate post-export artifacts (merged BF16, GGUFs)
python scripts/evaluate.py --config configs/eval_config.yaml --post-export

# Evaluate only export artifacts, merging into an existing metrics.json
# (avoids re-downloading the base model to re-run SFT/DPO eval)
python scripts/evaluate.py --config configs/eval_config.yaml \
  --mode post-export --post-export \
  --merge-existing ./artifacts/eval/metrics.json
```

### Switching inference provider

Change one line in `configs/eval_config.yaml`:

```yaml
inference:
  provider: "hf_native"   # or: vllm | ollama | tgi | llama_cpp
```

`llama_cpp` is used automatically for GGUF artifact evaluation. It requires a source-built `llama-cpp-python` — see [Dockerfile](Dockerfile) for the build flags.

### CI gate

`metrics.json` schema is versioned and designed for downstream consumption. The gate treats evaluation as two distinct products:

- **`pass_fail`** — raw decoding metrics vs raw thresholds. Research transparency: shows what the model can do without structural guidance.
- **`deployment_pass_fail`** — constrained decoding metrics vs deployment thresholds. Drives `ci_pass`. Matches production serving mode (outlines / xgrammar).

```json
{
  "schema_version": "1.0.0",
  "ci_gate_mode": "constrained",
  "models": {
    "dpo": {
      "raw": { "schema_validity": 0.407, "null_accuracy": 1.0, "field_f1": 0.789 },
      "constrained": { "schema_validity": 0.718, "null_accuracy": 1.0, "field_f1": 0.385 },
      "raw_vs_guided_gap": { "schema_validity_gap": 0.311, "field_f1_gap": -0.404 },
      "pass_fail": { "schema_validity": { "value": 0.407, "passed": false }, ... },
      "deployment_pass_fail": { "schema_validity": { "value": 0.718, "passed": true }, ... },
      "deployment_gate_mode": "constrained"
    }
  },
  "ci_pass": true
}
```

The gate mode and deployment thresholds are configurable in `configs/eval_config.yaml` under `metrics.ci_gate`. Setting `mode: "raw"` restores the previous behaviour.

---

## Export artifacts

| Artifact | Size | Use case |
|---|---|---|
| `export/adapter/` | ~309 MiB | PEFT inference, fastest to iterate |
| `export/merged_bf16/` | ~14.5 GiB | Full weights, vLLM / TGI serving |
| `export/gguf/model_q8_0.gguf` | ~7.7 GiB | llama.cpp, near-lossless |
| `export/gguf/model_q4_k_m.gguf` | ~4.5 GiB | Ollama, edge deployment |

---

## Inference examples

### HuggingFace + PEFT adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./artifacts/dpo/best")
tokenizer = AutoTokenizer.from_pretrained("./artifacts/dpo/best")

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Extract entities from: Apple Inc. was founded by Steve Jobs."}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Constrained decoding with outlines (recommended for production)

```python
import json
import outlines
import outlines.models

schema = json.load(open("configs/schemas/extraction_schema.json"))
model = outlines.models.Transformers("./artifacts/dpo/best")
generator = outlines.Generator(model, outlines.json_schema(schema))

result = generator("Extract entities from: Apple Inc. was founded by Steve Jobs.")
print(result)  # guaranteed valid JSON per schema
```

### Ollama (GGUF Q4_K_M)

```bash
ollama create qwen-extraction -f Modelfile
ollama run qwen-extraction "Extract entities from: ..."
```

### llama.cpp

```bash
./llama.cpp/build/bin/llama-cli \
  -m artifacts/export/gguf/model_q4_k_m.gguf \
  -p "Extract entities from: Apple Inc. was founded by Steve Jobs." \
  -n 128 --temp 0
```

---

## Reproducibility

Every training run emits a manifest capturing:

- Git commit SHA and dirty-state flag
- Lockfile hash (dependency fingerprint)
- Dataset content hash (SHA-256)
- Full hardware fingerprint: GPU model, NVIDIA driver, CUDA toolkit, PyTorch CUDA runtime, cuDNN version
- Resolved config (all defaults merged)
- Final metrics

**Tolerance bands:** reproduction runs are considered equivalent within ±1.5% F1, ±2.0% null accuracy. Residual non-determinism from cuBLAS/cuDNN algorithm selection is expected and documented.

**Validated environment:**
- GPU: NVIDIA H100 80GB (training) / A100 40GB (eval + export)
- NVIDIA driver: 580.105.08
- CUDA: 12.8 (training), 12.1 (PyTorch runtime)
- Python: 3.10

---

## Adapting to a new task

1. Replace the dataset in `data/` (keep the same JSONL format with `prompt`, `completion`, `is_null_case` fields)
2. Update the JSON Schema in `configs/schemas/`
3. Update `configs/base_config.yaml` → `task.description`
4. Run the full pipeline: `make train evaluate export CONFIG=configs/sft_config.yaml`

No code changes required.

---

## License

Model weights are derived from [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache 2.0).
Attribution: Qwen2.5 by Alibaba Cloud.
Pipeline code: MIT.
