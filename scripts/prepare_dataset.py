"""
Phase 0 — Dataset preparation.
Load raw examples, split, decontaminate, validate null fractions, hash, serialize.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.dataset_builder import build_and_save_dataset
from src.data.storage import build_storage
from src.manifest.run_manifest import create_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-data", required=True, help="Path to raw JSONL data file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    storage = build_storage(cfg["storage"])

    with open(args.raw_data) as f:
        raw_examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(raw_examples)} raw examples")
    stats = build_and_save_dataset(raw_examples, cfg, storage)

    print(json.dumps(stats, indent=2))

    # Persist dataset hash to config for manifest
    cfg["dataset"]["content_hash"] = stats["combined_hash"]
    with open(args.config, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"Dataset prepared. Combined hash: {stats['combined_hash']}")


if __name__ == "__main__":
    main()
