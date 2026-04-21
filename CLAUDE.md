Role and Context
You are building a Specialized Fine-Tuning System: an end-to-end pipeline that takes a base open-weights language model and produces a task-specialized variant with measurably superior performance on a narrow task — initially structured JSON extraction or tool-calling accuracy. The system must be generic enough that swapping tasks requires only a new dataset, evaluator, and config — not a rewrite.
The design philosophy is adversarial against silent failure modes: reproducibility breaks at the driver layer, specialized models over-fire on null inputs, serving-stack changes move numbers, shipped artifacts diverge from trained ones. Every requirement below exists to prevent a specific silent failure.
Primary Goal
A reproducible pipeline that trains, evaluates, and exports a specialized model demonstrably outperforming its base checkpoint, with evaluation gates wired for CI/CD integration and verified on the artifact that actually ships.
System Requirements
Software Independence

Training stack: transformers, peft, trl, datasets, accelerate, bitsandbytes, pinned via lockfile. No unpinned dependencies.
Model artifacts: Export LoRA/QLoRA adapters (safetensors), merged full-weight checkpoint (safetensors), and GGUF with at least Q4_K_M and Q8_0 quantizations. Serving decisions must remain reversible.
Inference abstraction: Evaluation harness runs against HF-native, vLLM, Ollama, or TGI via a single config change. No vendor SDKs imported at top level.
Experiment tracking: W&B, MLflow, or plain-filesystem manifest behind a tracker interface. Pipeline runs end-to-end with tracking disabled.
Compute portability: Containerized (pinned CUDA/PyTorch). Document working paths for 24 GB consumer GPU (QLoRA), single A100/H100, and multi-GPU via accelerate.
Data layer: Storage abstraction over local paths, S3-compatible object storage, and HF Hub. Content-hash every dataset snapshot.

Licensing Compliance
Base-model and dataset licenses propagate to fine-tuned weights, and terms vary materially across Llama, Qwen, Mistral, Gemma, and others — including commercial-use restrictions, attribution requirements, derivative-work rules, and acceptable-use policies. Before training begins:

Record the base model's license, any use restrictions, and any attribution requirements in the run manifest.
Confirm dataset licenses are compatible with the base model's license and with intended downstream use. Document every source explicitly.
Propagate licensing metadata into the model card for every exported artifact. A model card missing license terms is a delivery failure.

Hardware-Level Reproducibility
Run manifest must capture: NVIDIA driver version (nvidia-smi), CUDA Toolkit version (nvcc --version), PyTorch's CUDA runtime (torch.version.cuda), cuDNN version, GPU model and count, kernel version (uname -r). Document the validated driver/CUDA combination. Host-driver/container-runtime mismatches are the most common silent reproducibility failure.
Seeds captured for Python, NumPy, PyTorch, CUDA, with a documented note on residual GPU non-determinism (cuBLAS, cuDNN algorithm selection).
Pipeline Reproducibility
Every run emits a manifest: git commit SHA, lockfile hash, dataset hash, resolved config, hardware fingerprint, final metrics, licensing metadata. One-command reproduction: make train CONFIG=... takes a clean checkout to a trained, evaluated, exported model. Configs are versioned YAML/TOML, never code.
Reproducibility tolerance is explicit: each metric has a documented tolerance band (e.g., ±1.5% F1) within which a reproduction run is considered equivalent. "Within noise" is defined, not assumed.
Observability
Training and validation loss per step. For DPO: reward margin, chosen/rejected rewards, KL divergence per step. Sample generations logged at intervals so drift is visible mid-run. GPU utilization and memory captured for capacity planning.
Implementation Plan
Phase 0 — Task Definition, Tokenizer Audit, Baseline

Freeze the task contract: input format, output schema (JSON Schema or typed tool catalog), failure modes including the null case.
Audit the tokenizer against task-specific syntax: braces, brackets, quotes, commas, colons, escapes, newlines, indentation. Check for byte-level fallbacks. Document any added special tokens or chat-template modifications — these must be carried through adapter, merged weights, and GGUF identically. Tokenizer drift between training and serving silently destroys structured-output accuracy.
Define metrics. Extraction: schema validity, field-level P/R/F1, exact match, type correctness, null-extraction accuracy. Tool-calling: tool-selection accuracy, argument-name and argument-value accuracy, end-to-end call validity, null-call accuracy.
Dataset: 2,000–10,000 examples, 80/10/10 split. Test set frozen from day one. Decontaminate via hash and n-gram overlap.
Negative examples are mandatory. Target distribution should match expected production traffic, with 10–15% as a floor. Include null cases in all splits proportionally. Null-case metrics are first-class.
Document dataset provenance, licenses, and any synthetic generation (generating model, prompt, especially for negative examples).
Evaluate base model on frozen test set, reporting positive and null metrics separately. This is the baseline.

Phase 1 — Supervised Fine-Tuning (SFT)
TRL's SFTTrainer with LoRA or QLoRA. QLoRA default for memory efficiency.
Scaling-factor discipline: The LoRA update scales as α/r. Alpha must stay fixed across all experiments in a study so rank is the variable and the scaling factor is varied deliberately. If rsLoRA is used (scaling α/√r), document the choice explicitly — it is not interchangeable with standard LoRA.
Hyperparameter framing: The values below are community-default starting points, not authoritative settings. Treat them as the initial config for a small sweep, not as final hyperparameters. Optimal values depend on base model, task, and dataset characteristics.
Starting configuration:

LoRA rank 16–64 (default 32), alpha fixed at 32 across rank trials, dropout 0.05.
Target modules: all attention and MLP linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).
Learning rate 1e-4 to 2e-4, cosine schedule, 3% warmup.
1–3 epochs, effective batch 16–64 via gradient accumulation.
bf16 where supported, fp16 otherwise. Gradient checkpointing on.

Checkpoint regularly, evaluate each on validation (positive and null), promote the best — not the last — to Phase 2.
Phase 2 — Direct Preference Optimization (DPO)
TRL's DPOTrainer from the best SFT checkpoint.
Reference-model memory handling: Standard DPO loads two model copies and OOMs a 24 GB card for 7B+ models. The pipeline must default to pre-computed reference log-probabilities (precompute_ref_log_probs=True) on single-GPU configurations: run the SFT checkpoint over the preference dataset once, cache log-probs, load only the policy model during training. Document memory profile for both modes.
Preference dataset construction:

Generate 4–8 completions per prompt from the SFT model, including negative-case prompts. Null-case preference pairs (chosen: correct abstention; rejected: hallucinated call) are high-leverage — DPO is well-suited to correcting over-eagerness.
Rank via deterministic rubric where possible (schema validators, unit tests on tool calls). Fall back to LLM-as-judge or human review only when programmatic ranking is infeasible.
Target 1,000–3,000 pairs. Quality over quantity.
Hold out a preference validation split for reward-margin monitoring.

Starting configuration (community defaults, tune per task):

Beta 0.1 (sweep range 0.05–0.3).
Learning rate 5e-7 to 5e-6 — an order of magnitude below SFT. Most common misconfiguration.
1 epoch typical; watch for reward hacking.
Same LoRA config as SFT (including fixed alpha), initialized from SFT adapters.

Phase 3 — Evaluation, Regression Check, CI/CD Hooks

Compare base, post-SFT, post-DPO on the frozen test set, broken out by positive and null cases.
Both unconstrained and constrained decoding. Production structured-output serving typically uses outlines, instructor, or xgrammar; evaluation must match. Report raw metrics, guided metrics, and the raw-vs-guided gap for every model. A model that hits 99% only under constraint has learned something meaningfully different from one that doesn't need it.
General-capability regression slice (small MMLU, HellaSwag, or IFEval subset) against all three models. DPO silently harms general capability when preference data is narrow. Non-negotiable.
At least 20 qualitative triples spanning easy, hard, null, and failure-mode cases.
Emit metrics.json: machine-readable summary of every metric, every model, both decoding modes, with pass/fail flags against configurable thresholds. This is the CI/CD contract — downstream GitOps pipelines gate merges and deployments against it without parsing prose. Schema is versioned with the pipeline.

Phase 4 — Export and Quantization Verification
Exporting produces the shipped artifact and must be evaluated as such.

Export: adapters (safetensors), merged weights (safetensors), GGUF (Q8_0 and Q4_K_M minimum).
Re-verify every shipped artifact on the frozen test set. Merging 4-bit QLoRA adapters into BF16 can introduce weight-shifting; GGUF conversion and quantization compound precision loss. The numbers reported in the final report must be the numbers of the artifact that ships, not the adapter during training. Evaluate {adapter, merged BF16, GGUF Q8_0, GGUF Q4_K_M} under both decoding modes. Record results in the report and metrics.json. Adapter-passes-but-GGUF-fails is a delivery failure.
Publish to registry with a model card covering training data, hyperparameters, per-artifact evaluation, tokenizer modifications, licensing terms and attribution requirements, known limitations, intended use.
Minimal inference example per runtime (transformers, vLLM, Ollama, llama.cpp), including a constrained-decoding example so consumers don't silently deploy in raw mode.

Deliverables
A single repository containing: training pipeline (SFT and DPO with log-prob pre-computation), dataset preparation with negative-example generation and decontamination, independent tokenizer-audit script, evaluation harness with pluggable inference providers and metrics.json emission, export tooling with post-export verification wired into the same harness, containerized environment with documented driver/CUDA validation, model card template, and a final report covering baseline through every shipped artifact format — including the adapter-to-quantized degradation profile and licensing metadata.
Out of Scope
RLHF with an online reward model. Pretraining or continued pretraining. Multi-task training. Serving infrastructure beyond minimal inference examples.
Success Criteria
A reviewer can clone the repo on a machine matching the documented driver/CUDA profile, run one command, reproduce the trained model within documented tolerance bands, consume metrics.json to gate a deployment, and read a report that honestly characterizes what the model does well, what it does poorly — including the raw-vs-guided gap and adapter-to-quantized degradation — and the licensing terms governing its use.
