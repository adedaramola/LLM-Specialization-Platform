"""
Run manifest: captures hardware fingerprint, git state, dataset hash,
lockfile hash, config, and final metrics into a single JSON artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _run(cmd: list[str], default: str = "unavailable") -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return default


@dataclass
class HardwarefingerPrint:
    nvidia_driver: str
    cuda_toolkit: str
    pytorch_cuda: str
    cudnn_version: str
    gpu_model: str
    gpu_count: int
    kernel: str
    cpu: str
    ram_gb: float


@dataclass
class RunManifest:
    run_id: str
    timestamp_utc: str
    git_commit: str
    git_dirty: bool
    lockfile_hash: str
    dataset_hash: str
    config: dict[str, Any]
    hardware: HardwarefingerPrint
    licensing: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    reproduction_command: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hardware"] = asdict(self.hardware)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def capture_hardware() -> HardwarefingerPrint:
    nvidia_driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    cuda_toolkit = _run(["nvcc", "--version"])
    kernel = _run(["uname", "-r"])
    cpu = platform.processor() or _run(["uname", "-m"])

    try:
        import torch
        pytorch_cuda = torch.version.cuda or "cpu-only"
        cudnn_version = str(torch.backends.cudnn.version())
        gpu_count = torch.cuda.device_count()
        gpu_model = torch.cuda.get_device_name(0) if gpu_count > 0 else "none"
    except ImportError:
        pytorch_cuda = cudnn_version = "torch-not-installed"
        gpu_count = 0
        gpu_model = "none"

    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        ram_gb = -1.0

    return HardwarefingerPrint(
        nvidia_driver=nvidia_driver,
        cuda_toolkit=cuda_toolkit,
        pytorch_cuda=pytorch_cuda,
        cudnn_version=cudnn_version,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        kernel=kernel,
        cpu=cpu,
        ram_gb=ram_gb,
    )


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataset(paths: list[str | Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        if Path(p).exists():
            h.update(hash_file(p).encode())
    return h.hexdigest()


def get_git_info() -> tuple[str, bool]:
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty_output = _run(["git", "status", "--porcelain"])
    return commit, bool(dirty_output and dirty_output != "unavailable")


def create_manifest(
    config: dict[str, Any],
    dataset_paths: list[str | Path],
    lockfile_path: str | Path = "requirements.txt",
    run_id: str | None = None,
) -> RunManifest:
    import uuid
    from datetime import datetime, timezone

    run_id = run_id or str(uuid.uuid4())[:8]
    git_commit, git_dirty = get_git_info()
    lockfile_hash = hash_file(lockfile_path) if Path(lockfile_path).exists() else "missing"
    dataset_hash = hash_dataset(dataset_paths)

    licensing = {
        "base_model": config.get("model", {}).get("name", "unknown"),
        "license": config.get("model", {}).get("license", "unknown"),
        "restrictions": config.get("model", {}).get("license_restrictions", ""),
        "attribution": config.get("model", {}).get("attribution", ""),
        "dataset_sources": config.get("dataset", {}).get("sources", []),
        "dataset_licenses": config.get("dataset", {}).get("licenses", []),
    }

    return RunManifest(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        lockfile_hash=lockfile_hash,
        dataset_hash=dataset_hash,
        config=config,
        hardware=capture_hardware(),
        licensing=licensing,
        reproduction_command=f"make train CONFIG={config.get('_config_path', 'configs/sft_config.yaml')}",
    )
