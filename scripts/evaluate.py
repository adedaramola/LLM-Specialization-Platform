"""
Phase 3 — Evaluation harness entry point.
Evaluates base, SFT, DPO, and all exported artifacts.
Emits metrics.json for CI/CD gating.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.evaluation.harness import build_provider, collect_qualitative_samples, evaluate_model, emit_metrics_json
from src.evaluation.regression import run_regression, check_regression
from src.evaluation.report import generate_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="all",
                        choices=["baseline", "post-sft", "post-dpo", "all", "post-export"])
    parser.add_argument("--all-artifacts", action="store_true")
    parser.add_argument("--post-export", action="store_true")
    parser.add_argument("--merge-existing", metavar="PATH",
                        help="Load existing metrics.json and merge new results into it")
    args = parser.parse_args()

    cfg = load_config(args.config)

    schema_path = cfg["constrained"]["schema_path"]
    with open(schema_path) as f:
        schema = json.load(f)

    test_examples = _load_test(cfg["dataset"]["test_path"])

    results = []
    mode = args.mode

    # Seed results from an existing metrics.json so prior model evals aren't re-run
    if args.merge_existing and Path(args.merge_existing).exists():
        with open(args.merge_existing) as f:
            existing = json.load(f)
        for label, entry in existing.get("models", {}).items():
            results.append({
                "model": label,
                "model_path": entry.get("model_path", ""),
                "provider": entry.get("provider", ""),
                "raw": entry.get("raw", {}),
                "constrained": entry.get("constrained", {}),
                "raw_vs_guided_gap": entry.get("raw_vs_guided_gap", {}),
                "n_examples": entry.get("n_examples", 0),
                "n_positive": entry.get("n_positive", 0),
                "n_null": entry.get("n_null", 0),
            })
        print(f"Loaded {len(results)} existing model results from {args.merge_existing}")

    if args.all_artifacts or mode in ("baseline", "all"):
        base_cfg = cfg["models"]["base"]
        if base_cfg.get("path"):
            print(f"Evaluating base model: {base_cfg['path']}")
            r = evaluate_model(
                model_label="base",
                model_path=base_cfg["path"],
                test_examples=test_examples,
                schema=schema,
                eval_cfg=cfg,
            )
            results.append(r)
            _print_result(r)

    if args.all_artifacts or mode in ("post-sft", "all"):
        sft_cfg = cfg["models"]["sft"]
        print(f"Evaluating SFT: {sft_cfg['path']}")
        r = evaluate_model("sft", sft_cfg["path"], test_examples, schema, cfg)
        results.append(r)
        _print_result(r)

    if args.all_artifacts or mode in ("post-dpo", "all"):
        dpo_cfg = cfg["models"]["dpo"]
        print(f"Evaluating DPO: {dpo_cfg['path']}")
        r = evaluate_model("dpo", dpo_cfg["path"], test_examples, schema, cfg)
        results.append(r)
        _print_result(r)

        # Regression check
        base_result = next((x for x in results if x["model"] == "base"), None)
        if base_result:
            print("Running regression check...")
            provider = build_provider(
                cfg["inference"]["provider"],
                dpo_cfg["path"],
                cfg["inference"],
            )
            new_regression = run_regression(
                provider,
                cfg["generation"],
                cfg["regression"]["benchmarks"],
                cfg["regression"]["mmlu_subjects"],
                cfg["regression"]["num_samples"],
            )
            base_provider = build_provider(
                cfg["inference"]["provider"],
                base_result["model_path"],
                cfg["inference"],
            )
            base_regression = run_regression(
                base_provider,
                cfg["generation"],
                cfg["regression"]["benchmarks"],
                cfg["regression"]["mmlu_subjects"],
                cfg["regression"]["num_samples"],
            )
            reg_report = check_regression(
                base_regression, new_regression,
                cfg["metrics"]["regression_thresholds"],
            )
            print(f"Regression check passed: {reg_report['passed']}")
            if not reg_report["passed"]:
                print("REGRESSION DETECTED:", json.dumps(reg_report["deltas"], indent=2))

    if args.post_export or mode == "post-export":
        for artifact_key, artifact_cfg in cfg["export_artifacts"].items():
            path = artifact_cfg["path"]
            if not Path(path).exists() and not path.endswith(".gguf"):
                print(f"Skipping {artifact_key}: path not found ({path})")
                continue
            _free_gpu_memory()
            print(f"Evaluating artifact: {artifact_key}")
            orig_provider = cfg["inference"]["provider"]
            cfg["inference"]["provider"] = artifact_cfg["runtime"]
            r = evaluate_model(artifact_key, path, test_examples, schema, cfg)
            cfg["inference"]["provider"] = orig_provider
            results.append(r)
            _print_result(r)

    output = emit_metrics_json(
        results,
        cfg["metrics"]["thresholds"],
        cfg["output"]["metrics_json"],
        schema_version=cfg["output"].get("metrics_schema_version", "1.0.0"),
        gate_cfg=cfg["metrics"].get("ci_gate"),
    )

    print(f"\nmetrics.json written to: {cfg['output']['metrics_json']}")
    print(f"CI gate: {'PASS' if output['ci_pass'] else 'FAIL'}")

    # Qualitative samples — use the best available model (DPO > SFT > base)
    qual_cfg = cfg.get("qualitative", {})
    n_qual = qual_cfg.get("num_samples", 20)
    qual_out = qual_cfg.get("output_path", "./artifacts/eval/qualitative_samples.json")
    if n_qual > 0 and results:
        best_result = next(
            (r for r in reversed(results) if r["model"] in ("dpo", "sft") and r.get("model_path")),
            next((r for r in reversed(results) if r.get("model_path")), None),
        )
        if best_result is None:
            print("Skipping qualitative samples: no model with a valid path in results")
            n_qual = 0
        else:
            print(f"\nCollecting {n_qual} qualitative samples from model: {best_result['model']}")
            qual_samples = collect_qualitative_samples(
                model_label=best_result["model"],
                model_path=best_result["model_path"],
                test_examples=test_examples,
                schema=schema,
                eval_cfg=cfg,
                n_samples=n_qual,
                seed=cfg.get("reproducibility", {}).get("seed", 42),
            )
            Path(qual_out).parent.mkdir(parents=True, exist_ok=True)
            with open(qual_out, "w") as f:
                json.dump(qual_samples, f, indent=2, ensure_ascii=False)
            by_cat: dict[str, int] = {}
            for s in qual_samples:
                by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
            print(f"Qualitative samples written to: {qual_out}")
            print(f"  Category breakdown: {by_cat}")

    # Report generation — fill model card template with actual metrics
    report_path = cfg.get("output", {}).get("report_path")
    if report_path:
        dpo_manifest = Path(cfg.get("training", {}).get("output_dir", "./artifacts/dpo")) / "manifest.json"
        sft_manifest = Path("./artifacts/sft/manifest.json")
        manifest_path = str(dpo_manifest) if dpo_manifest.exists() else (
            str(sft_manifest) if sft_manifest.exists() else None
        )
        generate_report(
            template_path="templates/model_card.md",
            metrics_json_path=cfg["output"]["metrics_json"],
            manifest_path=manifest_path,
            config=cfg,
            output_path=report_path,
        )
        print(f"Report written to: {report_path}")

    if not output["ci_pass"]:
        sys.exit(1)


def _free_gpu_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _load_test(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _print_result(r: dict) -> None:
    print(f"  [{r['model']}] raw F1={r['raw'].get('field_f1', 0):.3f} "
          f"null_acc={r['raw'].get('null_accuracy', 0):.3f} "
          f"schema_valid={r['raw'].get('schema_validity', 0):.3f}")


if __name__ == "__main__":
    main()
