"""
Filter an existing preference_pairs.jsonl by minimum score margin.

Use this to quickly improve DPO signal quality from an already-generated
preference dataset without re-running the full generation pipeline.

Diagnosis: the original dataset had 41% of pairs with margin < 0.5 (chosen
and rejected scores differed by < 12.5%). DPO cannot distinguish these pairs
at the log-probability level, producing noisy or negative reward margins.

Usage:
    python scripts/filter_preferences.py \\
        --input  artifacts/dpo/preference_dataset/preference_pairs.jsonl \\
        --output artifacts/dpo/preference_dataset/preference_pairs_filtered.jsonl \\
        --min-margin 0.5

Then retrain DPO with:
    preference_cache pointing to the directory containing preference_pairs_filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Source preference_pairs.jsonl")
    parser.add_argument("--output", required=True, help="Filtered output path")
    parser.add_argument("--min-margin", type=float, default=0.5,
                        help="Minimum chosen_score - rejected_score to keep (default 0.5)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Val split fraction written alongside the train file")
    args = parser.parse_args()

    pairs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    margins_all = [p["chosen_score"] - p["rejected_score"] for p in pairs]
    kept = [p for p in pairs if p["chosen_score"] - p["rejected_score"] >= args.min_margin]
    margins_kept = [p["chosen_score"] - p["rejected_score"] for p in kept]

    null_frac_all  = sum(1 for p in pairs if p.get("is_null_case")) / len(pairs)
    null_frac_kept = sum(1 for p in kept  if p.get("is_null_case")) / len(kept) if kept else 0

    print(f"Input:  {len(pairs)} pairs  mean_margin={statistics.mean(margins_all):.3f}  null={null_frac_all:.1%}")
    print(f"Kept:   {len(kept)} pairs  mean_margin={statistics.mean(margins_kept):.3f}  null={null_frac_kept:.1%}  "
          f"({100*len(kept)/len(pairs):.0f}% retained)")

    # Train/val split
    import random
    rng = random.Random(42)
    shuffled = kept.copy()
    rng.shuffle(shuffled)
    val_n = max(1, int(len(shuffled) * args.val_fraction))
    val_pairs   = shuffled[:val_n]
    train_pairs = shuffled[val_n:]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = out_path.parent / (out_path.stem + "_val" + out_path.suffix)

    with open(out_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Written: {len(train_pairs)} train → {out_path}")
    print(f"         {len(val_pairs)} val   → {val_path}")


if __name__ == "__main__":
    main()
