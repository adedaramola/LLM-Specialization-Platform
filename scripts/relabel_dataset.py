"""
Relabel an existing raw dataset under the v2 labeling contract.

The v1 dataset's labels were generated without a defined naming convention:
~42% of entity names were descriptive labels invented by the generator and
~35% of values were not verbatim in the text. Those labels are unlearnable,
which capped field F1 at ~0.47 regardless of model quality.

This script keeps every input text (the text distribution is unchanged) and
regenerates only the labels under src/data/labeling.py's deterministic
contract. Each response is checked by the programmatic validator; failures
are retried with the validator errors fed back to the model.

Usage:
  python scripts/relabel_dataset.py \
      --raw data/raw/news_extraction.jsonl \
      --out data/raw/news_extraction_v2.jsonl \
      [--limit 40]        # pilot run on the first N examples
      [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import anthropic

from src.data.labeling import LABELING_RULES, PROMPT_VERSION, validate_labeled_output

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = f"""You label news-article snippets with structured entity-extraction annotations.

Given an input text, respond with a JSON object of exactly this shape:
{{
  "null_extraction": <boolean>,
  "entities": [
    {{"name": "...", "type": "...", "value": "..."}}
  ]
}}

{LABELING_RULES}
Return only valid JSON — no markdown, no explanation."""


def _strip_code_fence(text: str) -> str:
    """The model sometimes wraps output in ```json fences despite instructions."""
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :] if "\n" in text else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def relabel_one(client: anthropic.Anthropic, input_text: str) -> dict | None:
    """Return a validated output object, or None after MAX_ATTEMPTS failures."""
    messages = [{"role": "user", "content": f"Label this text:\n\n{input_text}"}]
    for _ in range(MAX_ATTEMPTS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
            )
        except anthropic.AuthenticationError:
            raise SystemExit(
                "ERROR: Anthropic API key rejected (401). Check ANTHROPIC_API_KEY in .env."
            )
        except anthropic.APIError:
            continue
        text = _strip_code_fence(response.content[0].text.strip())
        try:
            output = json.loads(text)
            validate_labeled_output(output, input_text)
            return {"null_extraction": output["null_extraction"], "entities": output.get("entities", [])}
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
            # Feed the validator errors back so the retry can fix them.
            messages = messages[:1] + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"Your labels violate the contract: {e}. "
                    "Fix every violation and return only the corrected JSON.",
                },
            ]
    return None


def main():
    parser = argparse.ArgumentParser(description="Relabel raw dataset under the v2 labeling contract")
    parser.add_argument("--raw", default="data/raw/news_extraction.jsonl")
    parser.add_argument("--out", default="data/raw/news_extraction_v2.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only relabel the first N examples (pilot)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.raw) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Relabeling {len(rows)} examples with {MODEL} (prompt v{PROMPT_VERSION}, {args.workers} workers)")

    client = anthropic.Anthropic()
    results: list[dict | None] = [None] * len(rows)
    done = 0
    lock = threading.Lock()

    def work(i: int) -> None:
        nonlocal done
        results[i] = relabel_one(client, rows[i]["input"])
        with lock:
            done += 1
            if done % 50 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, range(len(rows))))

    relabeled, failed, null_flips = [], 0, 0
    for row, output in zip(rows, results):
        if output is None:
            failed += 1
            continue
        if output["null_extraction"] != row["output"].get("null_extraction", False):
            null_flips += 1
        relabeled.append(
            {
                "input": row["input"],
                "output": output,
                "metadata": {
                    "generation_model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "relabeled_from": str(args.raw),
                },
            }
        )

    n = len(relabeled)
    print(f"\nRelabeled {n}/{len(rows)} ({failed} failed after {MAX_ATTEMPTS} attempts)")
    if n == 0:
        raise SystemExit("ERROR: no examples were successfully relabeled — nothing written.")
    n_null = sum(1 for r in relabeled if r["output"]["null_extraction"])
    n_entities = sum(len(r["output"]["entities"]) for r in relabeled)
    print(f"Null fraction: {n_null}/{n} ({n_null / n:.1%}), null-status flips vs v1: {null_flips}")
    print(f"Entities: {n_entities} ({n_entities / max(n - n_null, 1):.2f} per positive example)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in relabeled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Written to: {out_path}")
    print("\nNext step:")
    print(f"  python scripts/prepare_dataset.py --config configs/sft_config.yaml --raw-data {out_path}")


if __name__ == "__main__":
    main()
