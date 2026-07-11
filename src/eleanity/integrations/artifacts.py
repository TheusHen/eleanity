from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from eleanity.utils.logging import get_logger, log_event
from eleanity.utils.security import redact_mapping

logger = get_logger("eleanity.integrations")


class ArtifactSink(ABC):
    """Sink for run artifacts. Implementations must not upload raw prompts by default."""

    @abstractmethod
    def publish(self, run_dir: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]: ...


class LocalArtifactSink(ArtifactSink):
    """Copy redacted artifacts to a local directory (safe default)."""

    def __init__(self, destination: Path | str):
        self.destination = Path(destination)

    def publish(self, run_dir: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        run_dir = Path(run_dir)
        target = self.destination / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "results.sarif", "junit.xml", "github-annotations.txt", "summary.txt"):
            src = run_dir / name
            if src.is_file():
                if name == "result.json":
                    data = json.loads(src.read_text(encoding="utf-8"))
                    # Ensure share-safe export: keep structure, rely on existing redaction fields
                    (target / name).write_text(
                        json.dumps(redact_mapping(data), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                else:
                    shutil.copy2(src, target / name)
        if metadata:
            (target / "sink-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log_event(logger, "artifact_exported", sink="local", path=str(target))
        return {"sink": "local", "path": str(target)}


class MLflowArtifactSink(ArtifactSink):
    """Optional MLflow artifact sink — metadata + files only, no prompt requirement.

    Requires `mlflow` to be installed in the environment. Disabled unless used explicitly.
    """

    def __init__(self, experiment: str = "eleanity", run_name: str | None = None):
        self.experiment = experiment
        self.run_name = run_name

    def publish(self, run_dir: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError("mlflow is not installed") from error
        run_dir = Path(run_dir)
        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=self.run_name or run_dir.name):
            result_path = run_dir / "result.json"
            if result_path.is_file():
                data = json.loads(result_path.read_text(encoding="utf-8"))
                diagnosis = data.get("diagnosis") or {}
                mlflow.log_param("run_id", data.get("run_id"))
                mlflow.log_param("status", diagnosis.get("status"))
                mlflow.log_param("first_divergence", diagnosis.get("first_divergence"))
                if data.get("total_duration_ms") is not None:
                    mlflow.log_metric("total_duration_ms", float(data["total_duration_ms"]))
                # Log files but prefer SARIF/junit over full prompts
                for name in ("results.sarif", "junit.xml", "summary.txt"):
                    path = run_dir / name
                    if path.is_file():
                        mlflow.log_artifact(str(path))
                # Redacted result only
                redacted = redact_mapping(data)
                tmp = run_dir / "_redacted_result.json"
                tmp.write_text(json.dumps(redacted, indent=2), encoding="utf-8")
                mlflow.log_artifact(str(tmp))
                tmp.unlink(missing_ok=True)
            if metadata:
                for key, value in metadata.items():
                    mlflow.log_param(f"meta_{key}", str(value)[:250])
        log_event(logger, "artifact_exported", sink="mlflow", experiment=self.experiment)
        return {"sink": "mlflow", "experiment": self.experiment}


class WandbArtifactSink(ArtifactSink):
    """Optional Weights & Biases sink — redacted artifacts only."""

    def __init__(self, project: str = "eleanity", entity: str | None = None):
        self.project = project
        self.entity = entity

    def publish(self, run_dir: Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("wandb is not installed") from error
        run_dir = Path(run_dir)
        run = wandb.init(project=self.project, entity=self.entity, job_type="eleanity-parity", reinit=True)
        result_path = run_dir / "result.json"
        if result_path.is_file():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            diagnosis = data.get("diagnosis") or {}
            wandb.config.update(
                {
                    "run_id": data.get("run_id"),
                    "status": diagnosis.get("status"),
                    "first_divergence": diagnosis.get("first_divergence"),
                },
                allow_val_change=True,
            )
            artifact = wandb.Artifact(name=f"eleanity-{run_dir.name}", type="parity-report")
            for name in ("results.sarif", "junit.xml", "summary.txt"):
                path = run_dir / name
                if path.is_file():
                    artifact.add_file(str(path))
            redacted_path = run_dir / "_redacted_result.json"
            redacted_path.write_text(
                json.dumps(redact_mapping(data), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            artifact.add_file(str(redacted_path))
            run.log_artifact(artifact)
            redacted_path.unlink(missing_ok=True)
        run.finish()
        log_event(logger, "artifact_exported", sink="wandb", project=self.project)
        return {"sink": "wandb", "project": self.project}


def export_run_artifacts(
    run_dir: Path | str,
    sink: str = "local",
    *,
    destination: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if sink == "local":
        dest = destination or Path(".eleanity/exports")
        return LocalArtifactSink(dest).publish(run_dir, metadata=metadata)
    if sink == "mlflow":
        return MLflowArtifactSink().publish(run_dir, metadata=metadata)
    if sink in {"wandb", "wb"}:
        return WandbArtifactSink().publish(run_dir, metadata=metadata)
    raise ValueError(f"unknown sink: {sink} (local|mlflow|wandb)")
