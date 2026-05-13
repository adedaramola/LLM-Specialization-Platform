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

### Per-artifact evaluation

| Artifact | Schema Validity | Field F1 | Exact Match | Null Accuracy | Decoding |
|---|---|---|---|---|---|
| Merged BF16 (vLLM) | 0.987 | 0.466 | 0.165 | **1.000** | raw |
| Merged BF16 (vLLM) | 0.987 | **0.476** | 0.168 | **1.000** | constrained |
| GGUF Q8_0 | **0.997** | **0.486** | 0.157 | **1.000** | raw |
| GGUF Q4_K_M | 0.989 | 0.466 | 0.154 | **1.000** | raw |

**Field F1** measures field-name + value alignment on positive examples only (null cases excluded).
**Null accuracy** measures correct abstention on inputs with nothing to extract.

### General-capability regression (MMLU + HellaSwag slice)

| Benchmark | Base | DPO | Delta | Status |
|---|---|---|---|---|
| MMLU (102 examples) | 0.324 | 0.451 | **+0.127** | PASS |
| HellaSwag (100 examples) | 0.310 | 0.330 | **+0.020** | PASS |

DPO did not harm general capability. MMLU improved substantially — DPO sharpened instruction-following on the same Qwen ChatML format used by the task.

### CI gate result: PASS

| Metric | Value | Threshold |
|---|---|---|
| Schema validity (constrained) | 0.987 | ≥ 0.95 |
| Field F1 (constrained) | 0.476 | ≥ 0.45 |
| Null accuracy (constrained) | 1.000 | ≥ 0.95 |
| Exact match (constrained) | 0.168 | ≥ 0.10 |

### Key findings

- **Null accuracy is perfect across all artifacts** — the model never hallucinates an extraction when there is nothing to extract. This holds through BF16 merge and both GGUF quantizations.
- **Raw-vs-guided gap is near zero** — field F1 improves only +0.011 under constrained decoding. The model produces valid JSON structure without guidance at temperature 0, but constrained decoding is still recommended in production for the hard guarantee.
- **GGUF quantization degradation is minimal** — Q8_0 → Q4_K_M drops field F1 by 0.020 and null accuracy by 0. Q4_K_M is production-viable.
- **Field F1 ceiling is moderate (~0.47)** — the model correctly identifies entity types and values but field-name alignment with the reference schema is inconsistent. This reflects the training data distribution; improving it requires richer schema coverage in the dataset.

---

## Project structure

```
.
├── configs/
│   ├── base_config.yaml          # Shared defaults — all stage configs inherit this
│   ├── sft_config.yaml           # SFT hyperparameters
│   ├── dpo_config.yaml           # DPO hyperparameters + preference data settings
│   ├── eval_config.yaml          # Evaluation provider, thresholds, CI gate config
│   └── schemas/
│       └── extraction_schema.json
│
├── scripts/
│   ├── bootstrap_instance.sh     # Pre-flight check: AWS, data, GPU, imports — run first
│   ├── start_eval.sh             # Start vLLM + run eval in sequence, upload results to S3
│   ├── setup_instance.sh         # One-time venv + dependency setup (--train, --gguf flags)
│   ├── prepare_dataset.py        # Build + decontaminate dataset, content-hash snapshot
│   ├── tokenizer_audit.py        # Verify tokenizer handles task-specific syntax
│   ├── train_sft.py              # Phase 1: QLoRA SFT
│   ├── generate_preferences.py   # Build DPO preference pairs from SFT + base completions
│   ├── filter_preferences.py     # Filter existing preference dataset by min_margin
│   ├── train_dpo.py              # Phase 2: DPO from best SFT checkpoint
│   ├── evaluate.py               # Phase 3: eval harness, metrics.json, qualitative samples
│   └── export.py                 # Phase 4: adapter → merged BF16 → GGUF
│
├── src/
│   ├── data/
│   │   ├── dataset_builder.py
│   │   ├── decontamination.py
│   │   ├── preference_builder.py # min_margin filter + base model rejection support
│   │   └── storage.py            # Local / S3 / HF Hub abstraction
│   ├── training/
│   │   ├── sft_trainer.py
│   │   └── dpo_trainer.py        # Ref log-prob precomputation (single-GPU safe)
│   ├── evaluation/
│   │   ├── harness.py
│   │   ├── metrics.py
│   │   ├── regression.py         # MMLU + HellaSwag general-capability slice
│   │   ├── report.py             # Model card generation from metrics.json + manifests
│   │   └── providers/
│   │       ├── hf_provider.py    # HuggingFace native (explicit PEFT adapter loading)
│   │       ├── llamacpp_provider.py
│   │       ├── vllm_provider.py  # Batched + guided_json constrained decoding
│   │       ├── ollama_provider.py
│   │       └── tgi_provider.py
│   ├── export/
│   │   └── exporter.py
│   ├── manifest/
│   │   └── run_manifest.py       # Git, hardware, dataset hashes
│   ├── tokenizer/
│   │   └── audit.py
│   └── tracking/
│       └── tracker.py
│
├── templates/
│   └── model_card.md
├── .dockerignore
├── Dockerfile                    # CUDA 12.8 + cuDNN 9.19 base; INSTALL_TRAIN and INSTALL_GGUF build args
├── Makefile
├── requirements-eval.lock        # Pinned eval stack (vLLM, transformers, peft, …)
└── requirements-dev.txt
```

---

## Quickstart

### Prerequisites

- NVIDIA GPU with ≥24 GB VRAM (validated: A100 40GB SXM4)
- NVIDIA driver ≥ 520 (validated: 580.105.08)
- CUDA Toolkit 12.8

### Lambda Labs instance (recommended)

This is the validated path. Training and eval require a GPU instance; local Mac/Linux can run tests and linting only.

```bash
# 1. Clone and set up the environment (once per instance)
git clone https://github.com/adedaramola/LLM-Specialization-Platform
cd LLM-Specialization-Platform
bash scripts/setup_instance.sh --train --gguf

# 2. Activate venv (every session)
source ~/vllm-env/bin/activate

# 3. Pre-flight check — catches missing data, stale GPU processes, broken imports
bash scripts/bootstrap_instance.sh

# 4. Train (SFT → preference generation → DPO)
make train

# 5. Export artifacts (adapter → merged BF16 → GGUF)
make export

# 6. Evaluate (starts vLLM, runs eval, uploads metrics.json to S3)
bash scripts/start_eval.sh --merge-existing

# 7. GGUF eval
bash scripts/start_eval.sh --gguf --merge-existing
```

### Docker (on GPU Linux host)

```bash
# Full image: eval + training + GGUF stacks
make docker-build

# Run training (artifacts/ and data/ are mounted as volumes)
make docker-run
```

### Local (tests and linting only — no GPU required)

```bash
make setup
source .venv/bin/activate
make check          # lint + smoke tests
```

---

## Training configuration

### SFT (Phase 1)

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Method | QLoRA (4-bit NF4, double quantization) |
| LoRA rank | 32 |
| LoRA alpha | 32 (fixed — scaling factor α/r held constant across rank experiments) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 1e-4, cosine schedule, 3% warmup |
| Epochs | 2 |
| Effective batch size | 32 (4 per device × 8 gradient accumulation) |
| Precision | bf16 + gradient checkpointing |

### DPO (Phase 2)

| Parameter | Value |
|---|---|
| Beta | 0.1 |
| Learning rate | 5e-7 (order of magnitude below SFT — most common misconfiguration) |
| Epochs | 1 |
| Preference pairs | 1,400 (1,260 train / 140 val) |
| Null-case pair fraction | 20% |
| Average score margin | 1.58 |
| Rejected source | Base model completions (scores ~1.5–2.5 vs SFT ~3.5–4.0) |
| Ref log-probs | Precomputed — only policy model loaded during training (single-GPU safe) |
| Ranking strategy | Deterministic (schema validator + field scorer) |

> **Why base model rejection?** SFT-vs-SFT preference pairs cluster at score margins ~0.5, which is too small for DPO to distinguish at the log-probability level. Using the base model as the rejected source produces margins ~1.58, giving a clear and reliable training signal. This is configured via `use_base_for_rejected: true` in `configs/dpo_config.yaml`.

---

## Evaluation

The evaluation harness runs every model under both unconstrained and constrained decoding, against the same frozen test set used for baseline. It emits `artifacts/eval/metrics.json` — a versioned, machine-readable summary with pass/fail flags.

```bash
# Recommended: use start_eval.sh (handles vLLM start + S3 upload automatically)
bash scripts/start_eval.sh                        # post-export eval via vLLM
bash scripts/start_eval.sh --gguf                 # GGUF eval via llama-cpp-python
bash scripts/start_eval.sh --merge-existing       # add results to existing metrics.json

# Direct evaluate.py invocation
python scripts/evaluate.py --config configs/eval_config.yaml --mode post-export --post-export
python scripts/evaluate.py --config configs/eval_config.yaml --mode baseline
```

### Switching inference provider

Change one line in `configs/eval_config.yaml`:

```yaml
inference:
  provider: "vllm"      # or: hf_native | ollama | tgi | llama_cpp
```

`llama_cpp` is used automatically for GGUF artifact evaluation. It requires `llama-cpp-python` built against CUDA — installed by `setup_instance.sh --gguf`.

### Constrained decoding

For `vllm` provider, constrained decoding uses vLLM's native `guided_json` parameter — no separate library required. For `hf_native`, use outlines or xgrammar. Set `constrained.enabled: true` in `eval_config.yaml`.

### CI gate

`metrics.json` emits two pass/fail verdicts:

- **`pass_fail`** — raw decoding vs raw thresholds (research transparency)
- **`deployment_pass_fail`** — constrained decoding vs deployment thresholds (drives `ci_pass`)

```json
{
  "schema_version": "1.0.0",
  "ci_gate_mode": "constrained",
  "models": {
    "dpo": {
      "raw":        { "schema_validity": 0.987, "field_f1": 0.466, "null_accuracy": 1.0 },
      "constrained":{ "schema_validity": 0.987, "field_f1": 0.476, "null_accuracy": 1.0 },
      "raw_vs_guided_gap": { "field_f1_gap": 0.011 },
      "deployment_pass_fail": { "schema_validity": { "value": 0.987, "passed": true } }
    }
  },
  "ci_pass": true
}
```

---

## Export artifacts

| Artifact | Size | Use case |
|---|---|---|
| `export/adapter/` | ~309 MiB | PEFT inference, fast iteration |
| `export/merged_bf16/` | ~15 GiB | Full weights, vLLM / TGI serving |
| `export/gguf/model_q8_0.gguf` | ~7.6 GiB | llama.cpp, near-lossless |
| `export/gguf/model_q4_k_m.gguf` | ~4.4 GiB | Ollama, edge deployment |

---

## Inference examples

### vLLM with constrained decoding (recommended for production)

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
  --model artifacts/export/merged_bf16 --dtype bfloat16 --port 8000

# Query with guided_json
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/artifacts/export/merged_bf16",
    "prompt": "<|im_start|>system\nExtract structured data as JSON.<|im_end|>\n<|im_start|>user\nFlight UA123 departs Chicago at 14:30.<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 256,
    "temperature": 0.0,
    "guided_json": {"type": "object", "properties": {"flight_number": {"type": "string"}}}
  }'
```

### HuggingFace + PEFT adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, "artifacts/dpo/best")
tokenizer = AutoTokenizer.from_pretrained("artifacts/dpo/best")

messages = [
    {"role": "system", "content": "Extract structured data as JSON. If nothing to extract, output NO_EXTRACTION."},
    {"role": "user", "content": "Flight UA123 departs Chicago at 14:30 to New York."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### llama.cpp (GGUF Q4_K_M)

```bash
./llama.cpp/build/bin/llama-cli \
  -m artifacts/export/gguf/model_q4_k_m.gguf \
  -p "<|im_start|>system\nExtract structured data as JSON.<|im_end|>\n<|im_start|>user\nFlight UA123 departs Chicago at 14:30.<|im_end|>\n<|im_start|>assistant\n" \
  -n 256 --temp 0
```

---

## Reproducibility

Every training run emits a manifest capturing:

- Git commit SHA and dirty-state flag
- Lockfile hash (dependency fingerprint)
- Dataset content hash (SHA-256)
- Full hardware fingerprint: GPU model, NVIDIA driver, CUDA toolkit, PyTorch CUDA runtime, cuDNN version
- Resolved config (all defaults merged)
- Final metrics and licensing metadata

**Tolerance bands:** reproduction runs are considered equivalent within ±1.5% F1, ±2.0% null accuracy. Residual non-determinism from cuBLAS/cuDNN algorithm selection is expected and documented.

**Validated environment:**

| Component | Version |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| NVIDIA driver | 580.105.08 |
| CUDA Toolkit | 12.8 (nvcc) |
| PyTorch CUDA | 13.0 |
| cuDNN | 9.19.0 |
| Python | 3.10 |

---

## Adapting to a new task

1. Replace the dataset in `data/` (keep JSONL format with `prompt`, `completion`, `is_null_case` fields)
2. Update the JSON Schema in `configs/schemas/`
3. Update `task.description` in `configs/base_config.yaml`
4. Run: `make train && make export && bash scripts/start_eval.sh`

No code changes required.

---

## License

Model weights are derived from [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache 2.0).
Attribution: Qwen2.5 by Alibaba Cloud.
Pipeline code: MIT.
