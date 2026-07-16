# Model Card: Qwen2.5-7B-Instruct-specialized

## Model Summary

| Field | Value |
|-------|-------|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Task | Extract structured JSON from unstructured text |
| Training phases | SFT → DPO |
| Pipeline version | 1.0.0 |
| Run ID | — |
| Git commit | — |

---

## Licensing

| Field | Value |
|-------|-------|
| Base model license | apache-2.0 |
| Commercial use | Permitted |
| Attribution required | Qwen2.5 by Alibaba Cloud |
| Restrictions | Attribution required. No additional restrictions for commercial use. |
| Dataset licenses | N/A (synthetic) |

> **A model card missing license terms is a delivery failure.**

---

## Training Data

| Field | Value |
|-------|-------|
| Dataset sources | Synthetically generated |
| Dataset licenses | N/A (synthetic) |
| Train / Val / Test split | 2998 / 374 / 377 |
| Null-case fraction | 0.140 |
| Dataset hash (combined) | — |
| Synthetic examples | None |
| Decontamination | Hash + 8-gram overlap against frozen test set |

---

## Tokenizer

| Field | Value |
|-------|-------|
| Tokenizer class | — |
| Vocab size | — |
| Chat template | — |
| Added special tokens | — |
| Byte-fallback characters | — |

> Tokenizer modifications must be carried through adapter, merged weights, and GGUF identically.
> Tokenizer drift between training and serving silently destroys structured-output accuracy.

---

## Training Configuration

### SFT (Phase 1)

| Hyperparameter | Value |
|----------------|-------|
| LoRA rank | 32 |
| LoRA alpha | 32 (fixed across rank experiments) |
| LoRA dropout | 0.050 |
| rsLoRA | False |
| Target modules |  |
| Learning rate | 1.00e-04 |
| LR schedule | cosine |
| Epochs | 2 |
| Effective batch size | 32 |
| Precision | bf16 |
| Gradient checkpointing | True |

### DPO (Phase 2)

| Hyperparameter | Value |
|----------------|-------|
| Beta | 0.100 |
| Learning rate | — |
| Epochs | 1 |
| Preference pairs | — |
| Null-case pair fraction | 0.200 |
| Ref log-prob precomputation | True |
| Ranking strategy | deterministic |

---

## Evaluation Results

> All metrics reported on the **frozen test set** (never used during training).
> Both unconstrained (raw) and constrained decoding reported.

### Per-Artifact Results

| Artifact | Schema Validity | Field F1 | Exact Match | Null Accuracy | Decoding |
|----------|----------------|----------|-------------|---------------|----------|
| Base | 0.401 | 0.000 | 0.000 | 1.000 | raw |
| SFT adapter | 1.000 | 0.884 | 0.499 | 1.000 | raw |
| SFT adapter | 1.000 | 0.884 | — | — | constrained |
| DPO adapter | 1.000 | 0.885 | 0.496 | 1.000 | raw |
| DPO adapter | 1.000 | 0.885 | — | — | constrained |
| Merged BF16 | 1.000 | 0.885 | 0.493 | 1.000 | raw |
| GGUF Q8_0 | 1.000 | 0.892 | 0.520 | 1.000 | raw |
| GGUF Q4_K_M | 1.000 | 0.889 | 0.525 | 1.000 | raw |

### Adapter-to-Quantized Degradation Profile

| Step | Field F1 delta | Null Accuracy delta |
|------|---------------|---------------------|
| Adapter → Merged BF16 | -6.78e-04 | 0.000 |
| Merged BF16 → GGUF Q8_0 | 0.007 | 0.000 |
| GGUF Q8_0 → GGUF Q4_K_M | -0.003 | 0.000 |

### General-Capability Regression

| Benchmark | Base | SFT | DPO | Delta (DPO vs Base) |
|-----------|------|-----|-----|---------------------|
| MMLU (slice) | — | — | — | — |
| HellaSwag (slice) | — | — | — | — |

---

## Reproducibility

| Field | Value |
|-------|-------|
| Reproduction command | `make train CONFIG=configs/eval_config.yaml` |
| Lockfile hash | — |
| Dataset hash | — |
| Seed | 42 |
| F1 tolerance band | ±0.015 |
| Null accuracy tolerance | ±0.020 |
| GPU | — × — |
| NVIDIA driver | — |
| CUDA Toolkit | — |
| PyTorch CUDA | — |
| cuDNN | — |

> **Residual non-determinism:** cuBLAS/cuDNN algorithm selection may produce slight
> result variation even with fixed seeds. Reproduction is considered equivalent when
> metrics fall within the documented tolerance bands.

---

## Inference Examples

### transformers (HF native)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

prompt = "<|im_start|>system\nYou are a precise JSON extractor...<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### transformers + outlines (constrained — recommended for production)
```python
import outlines
import outlines.models
import json

schema = json.load(open("configs/schemas/extraction_schema.json"))
model = outlines.models.Transformers("Qwen/Qwen2.5-7B-Instruct")
generator = outlines.Generator(model, outlines.json_schema(schema))
result = generator(prompt)
print(result)  # guaranteed valid JSON per schema
```

### Ollama (GGUF Q4_K_M)
```bash
ollama run qwen2.5-7b-instruct "Extract entities from: ..."
```

### llama.cpp
```bash
./llama.cpp/build/bin/llama-cli -m artifacts/export/gguf/model_q4_k_m.gguf -p "..." -n 128
```

---

## Known Limitations

- Evaluated only on the training task distribution; real-world inputs may differ in ways not captured by the test set.
- Model may produce syntactically valid JSON that does not conform to the schema without constrained decoding. Production deployment should use outlines or equivalent.
- Field-level extraction accuracy is low on this task — the model identifies entities correctly but field name alignment with the reference schema is inconsistent. This is a dataset labeling issue, not a model capacity issue.
- Evaluated only on the task distribution used for training. Out-of-distribution inputs are untested.
- Adapter-to-quantized degradation profile: see Evaluation Results table above.

## Intended Use

Extract structured JSON from unstructured text

## Out of Scope

- General-purpose instruction following (use the base model for this).
- Tasks other than Extract structured JSON from unstructured text.
