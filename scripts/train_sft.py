"""Phase 1 — SFT training entry point."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.manifest.run_manifest import create_manifest
from src.tracking.tracker import build_tracker
from src.training.sft_trainer import run_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Merge base config if defaults key present
    if "defaults" in cfg:
        config_dir = Path(args.config).parent
        base_cfg = {}
        for default in cfg.pop("defaults"):
            base_file = config_dir / f"{default}.yaml"
            if base_file.exists():
                with open(base_file) as bf:
                    base_cfg.update(yaml.safe_load(bf) or {})
        base_cfg.update(cfg)
        cfg = base_cfg

    cfg["_config_path"] = args.config

    tracker = build_tracker(cfg.get("tracking", {}))

    manifest = create_manifest(
        config=cfg,
        dataset_paths=[cfg["dataset"]["train_path"], cfg["dataset"]["val_path"]],
    )

    print(f"Run ID: {manifest.run_id}")
    print(f"Git commit: {manifest.git_commit}")
    print(f"Dataset hash: {manifest.dataset_hash}")

    best_checkpoint = run_sft(cfg, tracker=tracker)

    manifest.metrics["sft_checkpoint"] = best_checkpoint
    manifest_path = Path(cfg["training"]["output_dir"]) / "manifest.json"
    manifest.save(manifest_path)

    print(f"SFT complete. Best checkpoint: {best_checkpoint}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
