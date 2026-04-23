# Model Card: {{model_name}}

## Model Summary

| Field | Value |
|-------|-------|
| Base model | {{base_model}} |
| Task | {{task}} |
| Training phases | SFT → DPO |
| Pipeline version | {{pipeline_version}} |
| Run ID | {{run_id}} |
| Git commit | {{git_commit}} |

---

## Licensing

| Field | Value |
|-------|-------|
| Base model license | {{base_model_license}} |
| Commercial use | {{commercial_use}} |
| Attribution required | {{attribution}} |
| Restrictions | {{restrictions}} |
| Dataset licenses | {{dataset_licenses}} |

> **A model card missing license terms is a delivery failure.**

---

## Training Data

| Field | Value |
|-------|-------|
| Dataset sources | {{dataset_sources}} |
| Dataset licenses | {{dataset_licenses}} |
| Train / Val / Test split | {{train_size}} / {{val_size}} / {{test_size}} |
| Null-case fraction | {{null_fraction}} |
| Dataset hash (combined) | {{dataset_hash}} |
| Synthetic examples | {{synthetic_generation_model}} |
| Decontamination | Hash + {{ngram_n}}-gram overlap against frozen test set |

---

## Tokenizer

| Field | Value |
|-------|-------|
| Tokenizer class | {{tokenizer_class}} |
| Vocab size | {{vocab_size}} |
| Chat template | {{chat_template_present}} |
| Added special tokens | {{added_tokens}} |
| Byte-fallback characters | {{byte_fallback_chars}} |

> Tokenizer modifications must be carried through adapter, merged weights, and GGUF identically.
> Tokenizer drift between training and serving silently destroys structured-output accuracy.

---

## Training Configuration

### SFT (Phase 1)

| Hyperparameter | Value |
|----------------|-------|
| LoRA rank | {{lora_rank}} |
| LoRA alpha | {{lora_alpha}} (fixed across rank experiments) |
| LoRA dropout | {{lora_dropout}} |
| rsLoRA | {{use_rslora}} |
| Target modules | {{target_modules}} |
| Learning rate | {{sft_lr}} |
| LR schedule | {{lr_scheduler}} |
| Epochs | {{sft_epochs}} |
| Effective batch size | {{effective_batch}} |
| Precision | {{precision}} |
| Gradient checkpointing | {{gradient_checkpointing}} |

### DPO (Phase 2)

| Hyperparameter | Value |
|----------------|-------|
| Beta | {{dpo_beta}} |
| Learning rate | {{dpo_lr}} |
| Epochs | {{dpo_epochs}} |
| Preference pairs | {{num_preference_pairs}} |
| Null-case pair fraction | {{null_pair_fraction}} |
| Ref log-prob precomputation | {{precompute_ref_log_probs}} |
| Ranking strategy | {{ranking_strategy}} |

---

## Evaluation Results

> All metrics reported on the **frozen test set** (never used during training).
> Both unconstrained (raw) and constrained decoding reported.

### Per-Artifact Results

| Artifact | Schema Validity | Field F1 | Exact Match | Null Accuracy | Decoding |
|----------|----------------|----------|-------------|---------------|----------|
| Base | {{base_schema_validity}} | {{base_field_f1}} | {{base_exact_match}} | {{base_null_accuracy}} | raw |
| SFT adapter | {{sft_schema_validity}} | {{sft_field_f1}} | {{sft_exact_match}} | {{sft_null_accuracy}} | raw |
| SFT adapter | {{sft_constrained_schema_validity}} | {{sft_constrained_field_f1}} | — | — | constrained |
| DPO adapter | {{dpo_schema_validity}} | {{dpo_field_f1}} | {{dpo_exact_match}} | {{dpo_null_accuracy}} | raw |
| DPO adapter | {{dpo_constrained_schema_validity}} | {{dpo_constrained_field_f1}} | — | — | constrained |
| Merged BF16 | {{merged_schema_validity}} | {{merged_field_f1}} | {{merged_exact_match}} | {{merged_null_accuracy}} | raw |
| GGUF Q8_0 | {{gguf_q8_schema_validity}} | {{gguf_q8_field_f1}} | {{gguf_q8_exact_match}} | {{gguf_q8_null_accuracy}} | raw |
| GGUF Q4_K_M | {{gguf_q4_schema_validity}} | {{gguf_q4_field_f1}} | {{gguf_q4_exact_match}} | {{gguf_q4_null_accuracy}} | raw |

### Adapter-to-Quantized Degradation Profile

| Step | Field F1 delta | Null Accuracy delta |
|------|---------------|---------------------|
| Adapter → Merged BF16 | {{adapter_to_merged_f1_delta}} | {{adapter_to_merged_null_delta}} |
| Merged BF16 → GGUF Q8_0 | {{merged_to_q8_f1_delta}} | {{merged_to_q8_null_delta}} |
| GGUF Q8_0 → GGUF Q4_K_M | {{q8_to_q4_f1_delta}} | {{q8_to_q4_null_delta}} |

### General-Capability Regression

| Benchmark | Base | SFT | DPO | Delta (DPO vs Base) |
|-----------|------|-----|-----|---------------------|
| MMLU (slice) | {{base_mmlu}} | {{sft_mmlu}} | {{dpo_mmlu}} | {{dpo_mmlu_delta}} |
| HellaSwag (slice) | {{base_hellaswag}} | {{sft_hellaswag}} | {{dpo_hellaswag}} | {{dpo_hellaswag_delta}} |

---

## Reproducibility

| Field | Value |
|-------|-------|
| Reproduction command | `make train CONFIG={{config_path}}` |
| Lockfile hash | {{lockfile_hash}} |
| Dataset hash | {{dataset_hash}} |
| Seed | {{seed}} |
| F1 tolerance band | ±{{f1_tolerance}} |
| Null accuracy tolerance | ±{{null_tolerance}} |
| GPU | {{gpu_model}} × {{gpu_count}} |
| NVIDIA driver | {{nvidia_driver}} |
| CUDA Toolkit | {{cuda_toolkit}} |
| PyTorch CUDA | {{pytorch_cuda}} |
| cuDNN | {{cudnn_version}} |

> **Residual non-determinism:** cuBLAS/cuDNN algorithm selection may produce slight
> result variation even with fixed seeds. Reproduction is considered equivalent when
> metrics fall within the documented tolerance bands.

---

## Inference Examples

### transformers (HF native)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("{{hf_repo}}", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{{hf_repo}}")

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
model = outlines.models.Transformers("{{hf_repo}}")
generator = outlines.Generator(model, outlines.json_schema(schema))
result = generator(prompt)
print(result)  # guaranteed valid JSON per schema
```

### Ollama (GGUF Q4_K_M)
```bash
ollama run {{ollama_model_tag}} "Extract entities from: ..."
```

### llama.cpp
```bash
./llama.cpp/build/bin/llama-cli -m artifacts/export/gguf/model_q4_k_m.gguf -p "..." -n 128
```

---

## Known Limitations

- {{limitation_1}}
- Model may produce syntactically valid JSON that does not conform to the schema without constrained decoding. Production deployment should use outlines or equivalent.
- Field-level extraction accuracy is low on this task — the model identifies entities correctly but field name alignment with the reference schema is inconsistent. This is a dataset labeling issue, not a model capacity issue.
- Evaluated only on the task distribution used for training. Out-of-distribution inputs are untested.
- Adapter-to-quantized degradation profile pending post-export verification.

## Intended Use

{{intended_use}}

## Out of Scope

- General-purpose instruction following (use the base model for this).
- Tasks other than {{task}}.
