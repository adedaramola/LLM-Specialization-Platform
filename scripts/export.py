"""Phase 4 — Export entry point."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.export.exporter import run_full_export
from src.manifest.run_manifest import create_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Override DPO/SFT checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    checkpoint = args.checkpoint or cfg["training"].get("sft_checkpoint") or \
                 str(Path(cfg["training"]["output_dir"]) / "best")

    print(f"Exporting from checkpoint: {checkpoint}")
    artifact_paths = run_full_export(cfg, checkpoint)

    print("Export complete:")
    for k, v in artifact_paths.items():
        print(f"  {k}: {v}")

    export_manifest = {
        "checkpoint": checkpoint,
        "artifacts": artifact_paths,
    }
    out_path = Path(cfg["storage"].get("local_root", "./artifacts")) / "export" / "export_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(export_manifest, f, indent=2)

    print(f"Export manifest: {out_path}")


if __name__ == "__main__":
    main()
