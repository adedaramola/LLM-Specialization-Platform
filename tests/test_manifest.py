"""Tests for run manifest capture."""
import json
import tempfile
from pathlib import Path

from src.manifest.run_manifest import (
    hash_file,
    hash_dataset,
    get_git_info,
    create_manifest,
    capture_hardware,
)


class TestHashFile:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert hash_file(f) == hash_file(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        assert hash_file(f1) != hash_file(f2)


class TestHashDataset:
    def test_missing_paths_ok(self):
        result = hash_dataset(["/nonexistent/path.jsonl"])
        assert isinstance(result, str)
        assert len(result) == 64


class TestHardwareFingerprint:
    def test_captures_fields(self):
        hw = capture_hardware()
        assert hasattr(hw, "kernel")
        assert hasattr(hw, "gpu_count")
        assert isinstance(hw.gpu_count, int)


class TestCreateManifest:
    def test_creates_manifest(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("torch==2.3.1\n")
        cfg = {
            "model": {"name": "test/model", "license": "apache-2.0",
                      "license_restrictions": "", "attribution": ""},
            "dataset": {"sources": [], "licenses": []},
            "_config_path": "configs/sft_config.yaml",
        }
        manifest = create_manifest(cfg, [], lockfile_path=str(req))
        assert manifest.run_id
        assert manifest.lockfile_hash != "missing"
        assert manifest.licensing["base_model"] == "test/model"

    def test_save_roundtrip(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("torch==2.3.1\n")
        cfg = {
            "model": {"name": "x", "license": "mit",
                      "license_restrictions": "", "attribution": ""},
            "dataset": {"sources": [], "licenses": []},
        }
        manifest = create_manifest(cfg, [], lockfile_path=str(req))
        out = tmp_path / "manifest.json"
        manifest.save(out)
        data = json.loads(out.read_text())
        assert data["run_id"] == manifest.run_id
