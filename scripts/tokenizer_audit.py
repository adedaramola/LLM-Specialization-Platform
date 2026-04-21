"""
Phase 0 — Tokenizer audit.
Verifies task-critical character coverage and roundtrip fidelity.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.tokenizer.audit import audit_tokenizer, save_audit_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/tokenizer_audit.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    print(f"Auditing tokenizer: {model_name}")

    report = audit_tokenizer(model_name)
    save_audit_report(report, args.output)

    print(f"Audit complete. Passed: {report.to_dict()['passed']}")
    if report.issues:
        print("ISSUES:")
        for issue in report.issues:
            print(f"  ERROR: {issue}")
    if report.warnings:
        print("WARNINGS:")
        for w in report.warnings:
            print(f"  WARN: {w}")
    if report.byte_fallback_chars:
        print(f"Byte-fallback chars: {report.byte_fallback_chars}")

    if report.issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
