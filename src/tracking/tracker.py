"""
Tracker interface — wraps W&B, MLflow, or plain filesystem.
All pipeline code calls Tracker methods; the backend is resolved from config.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTracker(ABC):
    @abstractmethod
    def start_run(self, run_name: str, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    @abstractmethod
    def log_text(self, key: str, text: str, step: int | None = None) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class FilesystemTracker(BaseTracker):
    def __init__(self, log_dir: str):
        self._dir = Path(log_dir)
        self._run_name: str = ""
        self._run_dir: Path | None = None
        self._metrics: list[dict] = []

    def start_run(self, run_name: str, config: dict[str, Any]) -> None:
        self._run_name = run_name
        self._run_dir = self._dir / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        with open(self._run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        entry = {"step": step, **metrics}
        self._metrics.append(entry)
        if self._run_dir:
            with open(self._run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        if self._run_dir:
            with open(self._run_dir / f"{key}.txt", "a") as f:
                f.write(f"step={step}\n{text}\n---\n")

    def finish(self) -> None:
        if self._run_dir and self._metrics:
            with open(self._run_dir / "metrics_summary.json", "w") as f:
                json.dump(self._metrics, f, indent=2)


class WandbTracker(BaseTracker):
    def __init__(self, project: str):
        import wandb  # lazy import — not top-level
        self._wandb = wandb
        self._project = project
        self._run = None

    def start_run(self, run_name: str, config: dict[str, Any]) -> None:
        self._run = self._wandb.init(project=self._project, name=run_name, config=config)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        self._wandb.log({key: self._wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def finish(self) -> None:
        if self._run:
            self._run.finish()


class MLflowTracker(BaseTracker):
    def __init__(self, tracking_uri: str | None = None):
        import mlflow  # lazy import
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def start_run(self, run_name: str, config: dict[str, Any]) -> None:
        self._mlflow.start_run(run_name=run_name)
        self._mlflow.log_params(
            {k: str(v)[:250] for k, v in config.items()}
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_text(self, key: str, text: str, step: int | None = None) -> None:
        self._mlflow.log_text(text, f"{key}_{step}.txt")

    def finish(self) -> None:
        self._mlflow.end_run()


class NullTracker(BaseTracker):
    """No-op tracker — used when tracking.disabled = true."""
    def start_run(self, run_name: str, config: dict[str, Any]) -> None: pass
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: pass
    def log_text(self, key: str, text: str, step: int | None = None) -> None: pass
    def finish(self) -> None: pass


def build_tracker(tracking_cfg: dict[str, Any]) -> BaseTracker:
    if tracking_cfg.get("disabled", False):
        return NullTracker()

    backend = tracking_cfg.get("backend", "filesystem")
    if backend == "wandb":
        return WandbTracker(project=tracking_cfg.get("project", "llm-specialization"))
    elif backend == "mlflow":
        return MLflowTracker(tracking_uri=tracking_cfg.get("tracking_uri"))
    else:
        return FilesystemTracker(log_dir=tracking_cfg.get("log_dir", "./runs"))
