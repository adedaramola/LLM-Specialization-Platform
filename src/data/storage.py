"""
Storage abstraction: local filesystem, S3-compatible, or HF Hub.
All data I/O in the pipeline goes through StorageBackend.
"""
from __future__ import annotations

import hashlib
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class StorageBackend(ABC):
    @abstractmethod
    def read_text(self, path: str) -> str: ...

    @abstractmethod
    def write_text(self, path: str, content: str) -> None: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]: ...

    def content_hash(self, path: str) -> str:
        data = self.read_bytes(path)
        return hashlib.sha256(data).hexdigest()


class LocalStorage(StorageBackend):
    def __init__(self, root: str = "."):
        self._root = Path(root)

    def _abs(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self._root / p

    def read_text(self, path: str) -> str:
        return self._abs(path).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_bytes(self, path: str) -> bytes:
        return self._abs(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._abs(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def exists(self, path: str) -> bool:
        return self._abs(path).exists()

    def list(self, prefix: str) -> list[str]:
        base = self._abs(prefix)
        if not base.exists():
            return []
        return [str(p) for p in base.rglob("*") if p.is_file()]


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = ""):
        import boto3  # lazy import
        self._s3 = boto3.client("s3")
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def _key(self, path: str) -> str:
        return f"{self._prefix}/{path}".lstrip("/")

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def write_text(self, path: str, content: str) -> None:
        self.write_bytes(path, content.encode("utf-8"))

    def read_bytes(self, path: str) -> bytes:
        obj = self._s3.get_object(Bucket=self._bucket, Key=self._key(path))
        return obj["Body"].read()

    def write_bytes(self, path: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=self._key(path), Body=data)

    def exists(self, path: str) -> bool:
        import botocore.exceptions
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(path))
            return True
        except botocore.exceptions.ClientError:
            return False

    def list(self, prefix: str) -> list[str]:
        resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=self._key(prefix))
        return [obj["Key"] for obj in resp.get("Contents", [])]


class HFHubStorage(StorageBackend):
    def __init__(self, repo_id: str, repo_type: str = "dataset"):
        from huggingface_hub import HfApi  # lazy import
        self._api = HfApi()
        self._repo_id = repo_id
        self._repo_type = repo_type

    def read_text(self, path: str) -> str:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(self._repo_id, path, repo_type=self._repo_type)
        return Path(local).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        self._api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=path,
            repo_id=self._repo_id,
            repo_type=self._repo_type,
        )

    def read_bytes(self, path: str) -> bytes:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(self._repo_id, path, repo_type=self._repo_type)
        return Path(local).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        self._api.upload_file(
            path_or_fileobj=data,
            path_in_repo=path,
            repo_id=self._repo_id,
            repo_type=self._repo_type,
        )

    def exists(self, path: str) -> bool:
        files = self._api.list_repo_files(self._repo_id, repo_type=self._repo_type)
        return path in files

    def list(self, prefix: str) -> list[str]:
        files = self._api.list_repo_files(self._repo_id, repo_type=self._repo_type)
        return [f for f in files if f.startswith(prefix)]


def build_storage(cfg: dict) -> StorageBackend:
    backend = cfg.get("backend", "local")
    if backend == "s3":
        return S3Storage(bucket=cfg["s3_bucket"], prefix=cfg.get("s3_prefix", ""))
    elif backend == "hf_hub":
        return HFHubStorage(repo_id=cfg["hf_repo"])
    else:
        return LocalStorage(root=cfg.get("local_root", "."))
