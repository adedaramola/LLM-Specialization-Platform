"""
Phase 4 — Export: LoRA adapters, merged BF16 weights, GGUF (Q8_0 + Q4_K_M).
Tokenizer modifications carried through all artifacts identically.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def export_adapter(
    checkpoint_path: str,
    output_dir: str,
) -> str:
    """Copy adapter safetensors to export directory."""
    src = Path(checkpoint_path)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        shutil.copy2(f, dst / f.name)
    return str(dst)


def merge_and_export(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
) -> str:
    """Merge QLoRA adapters into full BF16 weights."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def export_gguf(
    merged_model_dir: str,
    output_dir: str,
    llama_cpp_dir: str = "/opt/llama.cpp",
    quantizations: list[str] | None = None,
) -> dict[str, str]:
    """Convert merged BF16 to GGUF and quantize to Q8_0 and Q4_K_M."""
    if quantizations is None:
        quantizations = ["Q8_0", "Q4_K_M"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    f32_path = Path(output_dir) / "model_f32.gguf"

    subprocess.run(
        ["python", str(convert_script), merged_model_dir, "--outfile", str(f32_path)],
        check=True,
    )

    quantize_bin = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    outputs: dict[str, str] = {}

    for quant in quantizations:
        out_path = Path(output_dir) / f"model_{quant.lower()}.gguf"
        subprocess.run(
            [str(quantize_bin), str(f32_path), str(out_path), quant],
            check=True,
        )
        outputs[quant] = str(out_path)

    f32_path.unlink(missing_ok=True)
    return outputs


def run_full_export(cfg: dict[str, Any], sft_or_dpo_checkpoint: str) -> dict[str, str]:
    export_cfg = cfg.get("export", {})
    base_model = cfg["model"]["name"]
    artifact_root = Path(cfg["storage"].get("local_root", "./artifacts")) / "export"

    adapter_dir = str(artifact_root / "adapter")
    merged_dir = str(artifact_root / "merged_bf16")
    gguf_dir = str(artifact_root / "gguf")

    export_adapter(sft_or_dpo_checkpoint, adapter_dir)
    merge_and_export(base_model, sft_or_dpo_checkpoint, merged_dir)
    gguf_paths = export_gguf(
        merged_dir,
        gguf_dir,
        llama_cpp_dir=export_cfg.get("llama_cpp_dir", "/opt/llama.cpp"),
        quantizations=export_cfg.get("quantizations", ["Q8_0", "Q4_K_M"]),
    )

    return {
        "adapter": adapter_dir,
        "merged_bf16": merged_dir,
        **{f"gguf_{k.lower()}": v for k, v in gguf_paths.items()},
    }
