"""
Phase 0 — Synthetic dataset generation using Claude API.

Generates news-article snippets with entity-extraction labels.
Entity types: person, organization, location, metric, date, event.
~15% null cases (opinion/vague text with no extractable entities).

Output: JSONL where each line is:
  {"input": "<article snippet>", "output": {"null_extraction": bool, "entities": [...]}}

Usage:
  python scripts/generate_dataset.py \
      --total 3000 \
      --out data/raw/news_extraction.jsonl \
      [--null-fraction 0.15] \
      [--batch-size 50] \
      [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os

from dotenv import load_dotenv
load_dotenv()

import anthropic


def _check_api_key() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        sys.exit("ERROR: ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    if len(key) < 20:
        sys.exit("ERROR: ANTHROPIC_API_KEY looks invalid. Check your .env file.")
    # Print only a masked hint — never the real value
    masked = key[:8] + "..." + key[-4:]
    print(f"API key loaded: {masked}")

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 512

SYSTEM_PROMPT = """You generate synthetic news article snippets paired with structured entity-extraction labels for a fine-tuning dataset.

Each response must be a JSON object with exactly this schema:
{
  "input": "<a realistic 2-4 sentence news article snippet>",
  "output": {
    "null_extraction": <boolean>,
    "entities": [
      {"name": "<entity name>", "type": "<entity type>", "value": "<normalized value>"}
    ]
  }
}

Entity types you may use: person, organization, location, metric, date, event.

Rules:
- null_extraction must be true when the snippet contains no specific extractable entities (editorial opinions, vague commentary, philosophical musings).
- When null_extraction is false, entities must contain at least one item.
- Do NOT include the "confidence" field.
- Names should be realistic but fictional (no real living public figures for persons).
- Metrics must have units in the value (e.g., "4.2%", "$1.3B", "12,000 jobs").
- Dates use ISO-8601 format in the value field (e.g., "2024-03-15").
- Events are named occurrences (elections, summits, disasters, ceremonies).
- Return only valid JSON — no markdown, no explanation."""

POSITIVE_PROMPTS = [
    "Generate a snippet about a corporate acquisition with financial metrics.",
    "Generate a snippet about a political election result with candidate names and vote percentages.",
    "Generate a snippet about a natural disaster affecting a specific city or region.",
    "Generate a snippet about a tech company product launch with revenue figures.",
    "Generate a snippet about a sports championship event with team names and scores.",
    "Generate a snippet about a central bank interest rate decision with the new rate.",
    "Generate a snippet about a government official announcing a new policy.",
    "Generate a snippet about an international summit or diplomatic meeting.",
    "Generate a snippet about a court ruling with the judge's name and verdict.",
    "Generate a snippet about a company's quarterly earnings with revenue and profit figures.",
    "Generate a snippet about a scientific breakthrough attributed to a research institution.",
    "Generate a snippet about a trade deal between two countries with trade volume.",
    "Generate a snippet about a CEO stepping down and a replacement being named.",
    "Generate a snippet about a city winning a bid to host an international event.",
    "Generate a snippet about a startup funding round with the amount raised and lead investor.",
    "Generate a snippet about unemployment statistics from a specific country.",
    "Generate a snippet about a military alliance or defense agreement.",
    "Generate a snippet about a public health crisis with infection numbers.",
    "Generate a snippet about an environmental regulation passed by a legislature.",
    "Generate a snippet about a celebrity or public figure's philanthropic initiative.",
    "Generate a snippet about a merger between two organizations with their headquarters locations.",
    "Generate a snippet about an infrastructure project (bridge, railway, highway) with cost.",
    "Generate a snippet about an award ceremony naming winners in multiple categories.",
    "Generate a snippet about a recall of consumer products by a manufacturer.",
    "Generate a snippet about immigration statistics announced by a government ministry.",
]

NULL_PROMPTS = [
    "Generate an editorial opinion snippet about the general state of modern journalism with no specific entities.",
    "Generate a philosophical commentary about the nature of democracy — vague, no named people or places.",
    "Generate a vague think-piece about economic inequality with no specific figures, countries, or organizations.",
    "Generate a cultural commentary about social media's impact on society — no specific platforms or statistics.",
    "Generate a short opinion piece about the importance of education in the 21st century with no specific data.",
    "Generate a snippet that expresses a general sentiment about political polarization without naming parties, people, or events.",
    "Generate an abstract reflection on climate change awareness without specific metrics, countries, or agreements.",
    "Generate a vague commentary on the future of work and automation without specific companies or statistics.",
]


def generate_example(client: anthropic.Anthropic, user_prompt: str) -> dict | None:
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
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()
        example = json.loads(text)
        _validate_example(example)
        return example
    except (json.JSONDecodeError, KeyError, ValueError):
        return None
    except anthropic.RateLimitError:
        time.sleep(10)
        return None
    except anthropic.APIError:
        return None


def _validate_example(example: dict) -> None:
    assert "input" in example and isinstance(example["input"], str)
    assert "output" in example
    out = example["output"]
    assert "null_extraction" in out and isinstance(out["null_extraction"], bool)
    if not out["null_extraction"]:
        assert "entities" in out and len(out["entities"]) >= 1
        for ent in out["entities"]:
            assert "name" in ent and "type" in ent and "value" in ent


def generate_batch(
    client: anthropic.Anthropic,
    prompts: list[str],
    batch_size: int,
) -> list[dict]:
    results = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        for prompt in chunk:
            example = generate_example(client, prompt)
            if example is not None:
                results.append(example)
        print(f"  Progress: {min(i + batch_size, len(prompts))}/{len(prompts)} requests, {len(results)} valid")
    return results


def build_prompt_list(total: int, null_fraction: float, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    n_null = int(total * null_fraction)
    n_positive = total - n_null

    # Allow more requests than target to account for ~10-20% parse failures
    oversample = 1.25
    pos_prompts = [rng.choice(POSITIVE_PROMPTS) for _ in range(int(n_positive * oversample))]
    null_prompts = [rng.choice(NULL_PROMPTS) for _ in range(int(n_null * oversample))]

    all_prompts = pos_prompts + null_prompts
    rng.shuffle(all_prompts)
    return all_prompts


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic news entity-extraction dataset")
    parser.add_argument("--total", type=int, default=3000, help="Target number of examples")
    parser.add_argument("--out", default="data/raw/news_extraction.jsonl", help="Output JSONL path")
    parser.add_argument("--null-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=50, help="Requests per progress update")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print 3 examples without writing")
    args = parser.parse_args()

    _check_api_key()
    client = anthropic.Anthropic()

    if args.dry_run:
        print("=== DRY RUN — generating 3 examples ===")
        prompts = build_prompt_list(10, args.null_fraction, args.seed)[:3]
        for i, prompt in enumerate(prompts):
            print(f"\n--- Example {i+1} (prompt: {prompt[:60]}...) ---")
            ex = generate_example(client, prompt)
            if ex:
                print(json.dumps(ex, indent=2, ensure_ascii=False))
            else:
                print("(failed to parse)")
        return

    print(f"Generating ~{args.total} examples ({args.null_fraction:.0%} null cases) using {MODEL}")
    prompts = build_prompt_list(args.total, args.null_fraction, args.seed)

    examples = generate_batch(client, prompts, args.batch_size)

    # Trim to target, preserving null fraction
    null_ex = [e for e in examples if e["output"]["null_extraction"]]
    pos_ex  = [e for e in examples if not e["output"]["null_extraction"]]

    n_null_target = int(args.total * args.null_fraction)
    n_pos_target  = args.total - n_null_target

    rng = random.Random(args.seed)
    null_final = rng.sample(null_ex, min(n_null_target, len(null_ex)))
    pos_final  = rng.sample(pos_ex,  min(n_pos_target,  len(pos_ex)))
    final = null_final + pos_final
    rng.shuffle(final)

    actual_null_frac = len(null_final) / len(final) if final else 0
    print(f"\nGenerated {len(final)} examples ({len(null_final)} null = {actual_null_frac:.1%})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Written to: {out_path}")
    print(f"\nNext step:")
    print(f"  python scripts/prepare_dataset.py --config configs/sft_config.yaml --raw-data {out_path}")


if __name__ == "__main__":
    main()
