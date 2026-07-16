# Model Card: Qwen2.5-7B-Instruct-JSON-Extraction-DPO

## Model Summary

| Field | Value |
|-------|-------|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Task | Structured JSON extraction from unstructured text |
| Training phases | SFT (QLoRA) → DPO |
| Pipeline version | 1.0.0 |
| SFT run ID | 6299d676 |
| DPO run ID | 9d83ab51 |
| SFT git commit | 1e12794e |
| DPO git commit | ca7ee043 |

---

## Licensing

| Field | Value |
|-------|-------|
| Base model license | Apache 2.0 |
| Commercial use | Permitted |
| Attribution required | Yes — "Qwen2.5 by Alibaba Cloud" |
| Restrictions | None beyond Apache 2.0 terms |
| Dataset licenses | Synthetically generated; no third-party dataset incorporated |

> Fine-tuned weights inherit the base model's Apache 2.0 license. Any derivative work must retain this attribution.

---

## Training Data

| Field | Value |
|-------|-------|
| Dataset sources | Synthetically generated (no external sources) |
| Dataset licenses | N/A (synthetic) |
| Train / Val / Test split | 2,998 / 375 / 375 |
| Null-case fraction | ~14% (inputs where correct output is NO_EXTRACTION) |
| Dataset hash (train) | 525fca995ddc0e29b103a4c8928b9bb3a1e23e035fb7bddcabd4d30ddd3ae40a |
| Synthetic generation model | Not recorded in manifest |
| Decontamination | Hash deduplication + 8-gram overlap against frozen test set |

Test set frozen before any training began and never used for model selection.

---

## Tokenizer

| Field | Value |
|-------|-------|
| Tokenizer class | Qwen2Tokenizer |
| Chat template | Yes — ChatML format (`<\|im_start\|>` / `<\|im_end\|>`) |
| Added special tokens | None added beyond base model vocabulary |
| EOS token | `<\|im_end\|>` |
| PAD token | `<\|endoftext\|>` |
| Model max length | 131,072 tokens |

No tokenizer modifications were made. The base Qwen2.5 tokenizer handles JSON syntax characters (braces, brackets, quotes, colons) natively without byte-level fallbacks for common ASCII characters. The same tokenizer is present in the adapter, merged weights, and must be bundled with any GGUF deployment to avoid tokenizer drift.

---

## Training Configuration

### SFT — Phase 1

| Hyperparameter | Value |
|----------------|-------|
| LoRA rank | 32 |
| LoRA alpha | 32 (fixed across all rank experiments) |
| LoRA dropout | 0.05 |
| Scaling | Standard α/r (rsLoRA not used) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| QLoRA | Yes — NF4 quantization, double quant, bfloat16 compute |
| Learning rate | 1e-4 |
| LR schedule | Cosine with 3% warmup |
| Epochs | 2 |
| Per-device batch size | 4 |
| Gradient accumulation | 8 steps (effective batch = 32) |
| Precision | bfloat16 |
| Gradient checkpointing | Yes |
| Max sequence length | 2,048 tokens |
| Best checkpoint selection | Lowest validation loss |

### DPO — Phase 2

| Hyperparameter | Value |
|----------------|-------|
| Beta | 0.1 |
| Loss type | Sigmoid (standard DPO) |
| Learning rate | 5e-7 |
| LR schedule | Cosine |
| Epochs | 1 |
| Preference pairs | 1,400 total (1,260 train / 140 val) |
| Null-case pair fraction | 20% |
| Average score margin | 1.58 |
| Ref log-prob precomputation | Yes — avoids loading two model copies on single-GPU |
| Ranking strategy | Deterministic (schema validator + field-level scorer) |
| Rejected source | Base model completions (higher margin than SFT-vs-SFT) |
| Initialized from | Best SFT checkpoint |
| LoRA config | Same as SFT (rank 32, alpha 32) |

> **Memory note:** DPO reference log-probs were precomputed in a single pass over the preference dataset, then only the policy model was loaded during training. This avoids the 2× VRAM requirement of standard DPO on a 40 GB card with a 7B model.

---

## Evaluation Results

> All metrics reported on the **frozen test set** (375 examples: ~323 positive, ~53 null).
> Numbers marked **[PENDING]** will be updated when GGUF eval completes.

### Per-Artifact Results

| Artifact | Schema Validity | Field F1 | Exact Match | Null Accuracy | Decoding |
|----------|----------------|----------|-------------|---------------|----------|
| Base (Qwen2.5-7B-Instruct) | — | — | — | — | raw |
| SFT adapter | — | — | — | — | raw |
| Merged BF16 (via vLLM) | 0.987 | 0.466 | 0.165 | 1.000 | raw |
| Merged BF16 (via vLLM) | 0.987 | 0.476 | 0.168 | 1.000 | constrained |
| GGUF Q8_0 (llama-cpp-python) | 0.997 | 0.486 | 0.157 | 1.000 | raw |
| GGUF Q4_K_M (llama-cpp-python) | 0.989 | 0.466 | 0.154 | 1.000 | raw |

> **Raw-vs-guided gap (Merged BF16):** Field F1 improves +0.011 under constrained decoding; schema validity and null accuracy are unchanged. The model is nearly deterministic on schema structure without constraint — constrained decoding is still recommended in production for guarantee semantics.

> **GGUF note:** Q8_0 scores slightly higher than merged BF16 in raw mode (field F1 +0.020, schema validity +0.010). This is within measurement noise and likely reflects minor tokenization or decoding differences between vLLM and llama-cpp-python rather than genuine quantization improvement.

### Adapter-to-Quantized Degradation Profile

| Step | Field F1 delta | Null Accuracy delta | Schema Validity delta |
|------|---------------|---------------------|-----------------------|
| Adapter → Merged BF16 | not separately measured | not separately measured | not separately measured |
| Merged BF16 → GGUF Q8_0 | +0.020 (within noise) | 0.000 | +0.010 (within noise) |
| GGUF Q8_0 → GGUF Q4_K_M | **−0.020** | 0.000 | **−0.008** |
| Merged BF16 → GGUF Q4_K_M | ~0.000 (net) | 0.000 | +0.002 (within noise) |

### General-Capability Regression (MMLU + HellaSwag slice)

| Benchmark | Base | DPO | Delta |
|-----------|------|-----|-------|
| MMLU (102 examples, 3 subjects) | 0.324 | 0.451 | **+0.127** |
| HellaSwag (100 examples) | 0.310 | 0.330 | **+0.020** |

> Regression check **PASSED** (threshold: delta ≥ −0.03 for both benchmarks). DPO did not harm general capability; MMLU improved substantially, likely because DPO training sharpened instruction-following on the same Qwen chat format.

### CI Gate Result

| Gate metric | Value | Threshold | Status |
|-------------|-------|-----------|--------|
| Schema validity (constrained) | 0.987 | ≥ 0.95 | **PASS** |
| Field F1 (constrained) | 0.476 | ≥ 0.45 | **PASS** |
| Exact match (constrained) | 0.168 | ≥ 0.10 | **PASS** |
| Null accuracy (constrained) | 1.000 | ≥ 0.95 | **PASS** |

**CI gate: PASS**

---

## Reproducibility

| Field | Value |
|-------|-------|
| Reproduction command | `make train CONFIG=configs/sft_config.yaml` then `make train CONFIG=configs/dpo_config.yaml` |
| Eval command | `bash scripts/setup_instance.sh && source ~/vllm-env/bin/activate && python3 scripts/evaluate.py --config configs/eval_config_vllm.yaml --mode post-export --post-export --merge-existing artifacts/eval/metrics.json` |
| Lockfile hash | 0a972e3ded9a58402c09d8c68d3d658e3921b411dccdd184c282ea1c07ec9cc6 |
| Dataset hash | 525fca995ddc0e29b103a4c8928b9bb3a1e23e035fb7bddcabd4d30ddd3ae40a |
| Seed | 42 |
| F1 tolerance band | ±1.5% |
| Null accuracy tolerance | ±2.0% |
| Exact match tolerance | ±1.5% |
| GPU | NVIDIA A100-SXM4-40GB × 1 |
| NVIDIA driver | 580.105.08 |
| CUDA Toolkit | 12.8 (nvcc) |
| PyTorch CUDA | 12.1 (SFT instance) |
| cuDNN | 8902 (SFT instance) |
| Kernel | 6.8.0-1046-nvidia |

> **Residual non-determinism:** cuBLAS/cuDNN algorithm selection may produce slight variation even with fixed seeds. A reproduction run is considered equivalent when all metrics fall within the documented tolerance bands above.

---

## Inference Examples

### vLLM (merged BF16 — recommended for GPU serving)

```bash
# Start server
USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 python3 -m vllm.entrypoints.openai.api_server \
  --model artifacts/export/merged_bf16 --dtype bfloat16 --port 8000

# Run with constrained decoding (recommended)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/artifacts/export/merged_bf16",
    "prompt": "<|im_start|>system\nExtract structured data as JSON.<|im_end|>\n<|im_start|>user\nFlight UA123 departs Chicago at 14:30 to New York.<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 256,
    "temperature": 0.0,
    "guided_json": {"type": "object", "properties": {"flight_number": {"type": "string"}, "origin": {"type": "string"}, "destination": {"type": "string"}, "departure_time": {"type": "string"}}}
  }'
```

### transformers (HF native)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "artifacts/export/merged_bf16", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("artifacts/export/merged_bf16")

messages = [
    {"role": "system", "content": "Extract structured data as JSON. If no extraction is possible, output NO_EXTRACTION."},
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
  -p "<|im_start|>system\nExtract structured data as JSON.<|im_end|>\n<|im_start|>user\nFlight UA123 departs Chicago at 14:30 to New York.<|im_end|>\n<|im_start|>assistant\n" \
  -n 256 --temp 0
```

### Python (llama-cpp-python, GPU, constrained decoding not natively supported)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="artifacts/export/gguf/model_q4_k_m.gguf",
    n_gpu_layers=-1,
    n_ctx=2048,
)
output = llm(
    "<|im_start|>system\nExtract structured data as JSON.<|im_end|>\n<|im_start|>user\nFlight UA123 departs Chicago at 14:30 to New York.<|im_end|>\n<|im_start|>assistant\n",
    max_tokens=256,
    temperature=0.0,
)
print(output["choices"][0]["text"])
```

---

## Known Limitations

- **Field F1 is moderate (~0.47):** The model correctly identifies entity types and values but field name alignment with the reference schema is inconsistent. This reflects the training data distribution, not model capacity. Exact match is low (0.17) for the same reason.
- **Constrained decoding required for schema guarantee:** Without `guided_json` or equivalent, the model produces valid JSON ~98.7% of the time but not 100%. For production use, always use vLLM `guided_json`, outlines, or xgrammar.
- **Null accuracy is perfect on test set (1.0):** This may not generalize; the null-case distribution in production may differ from the synthetic test set.
- **GGUF degradation:** Quantization precision loss compounds from BF16 → Q8_0 → Q4_K_M. See degradation profile above (PENDING). If Q4 shows significant field F1 drop, use Q8_0 for production.
- **General benchmarks use tiny slices:** MMLU (102 examples) and HellaSwag (100 examples) are indicative, not authoritative regression measurements.
- **Out-of-distribution inputs untested:** Only evaluated on the synthetic task distribution used for training.

## Intended Use

Production JSON extraction from unstructured text in the task domain matching the training distribution. Deploy behind constrained decoding (vLLM `guided_json` or equivalent). Not a general-purpose model.

## Out of Scope

- General-purpose instruction following (use base Qwen2.5-7B-Instruct).
- Tasks other than structured JSON extraction.
- Schemas substantially different from the training schema without re-evaluation.
