"""Phase 2 — DPO training entry point."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.manifest.run_manifest import create_manifest
from src.tracking.tracker import build_tracker
from src.training.dpo_trainer import precompute_reference_log_probs, run_dpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip ref log-prob precomputation (A100/H100 with enough VRAM)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    tracker = build_tracker(cfg.get("tracking", {}))

    if not args.skip_precompute and cfg["dpo"].get("precompute_ref_log_probs", True):
        print("Pre-computing reference log-probabilities (single-GPU memory optimization)...")
        pref_data_path = cfg["preference_data"]["preference_cache"]
        pref_data = _load_pref_data(pref_data_path)
        precompute_reference_log_probs(
            sft_checkpoint=cfg["training"]["sft_checkpoint"],
            preference_dataset=pref_data,
            cache_path=cfg["dpo"]["ref_log_probs_cache"],
            cfg=cfg,
        )
        print(f"Ref log-probs cached at: {cfg['dpo']['ref_log_probs_cache']}")

    pref_cache = cfg["preference_data"]["preference_cache"]
    manifest = create_manifest(
        config=cfg,
        dataset_paths=[
            str(Path(pref_cache) / "preference_pairs.jsonl"),
            str(Path(pref_cache) / "preference_pairs_val.jsonl"),
        ],
    )

    best_checkpoint = run_dpo(cfg, tracker=tracker)

    manifest.metrics["dpo_checkpoint"] = best_checkpoint
    manifest_path = Path(cfg["training"]["output_dir"]) / "manifest.json"
    manifest.save(manifest_path)

    print(f"DPO complete. Best checkpoint: {best_checkpoint}")


def _load_pref_data(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        data = []
        for f in p.glob("*.jsonl"):
            with open(f) as fh:
                data.extend(json.loads(l) for l in fh if l.strip())
        return data
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    main()
