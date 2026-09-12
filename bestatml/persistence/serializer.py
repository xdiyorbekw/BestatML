from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from bestatml.exceptions import BestatMLPersistenceError
from bestatml.version import __version__

ARTIFACT_FORMAT_VERSION = 1


def save_model(model: Any, path: str | Path) -> None:
    """Serialize a fitted BestatML estimator with versioned metadata."""
    target = Path(path).expanduser()
    if target.exists() and target.is_dir():
        raise BestatMLPersistenceError(
            f"BestatML[PersistenceError]: Cannot save an artifact to directory: {target}"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "library_version": __version__,
            "estimator_class": type(model).__name__,
            "model": model,
        }
        joblib.dump(payload, target, compress=3)
    except (OSError, ValueError, TypeError) as exc:
        raise BestatMLPersistenceError(
            f"BestatML[PersistenceError]: Could not save the model artifact.\n\n"
            f"Artifact:\n    {target}\n\nProblem:\n    {exc}\n\n"
            "How to fix:\n    Check the destination path, permissions, and available disk space."
        ) from exc


def load_model(cls: type, path: str | Path) -> Any:
    """Load a trusted BestatML artifact and validate its format and estimator type."""
    target = Path(path).expanduser()
    if not target.exists():
        raise BestatMLPersistenceError(
            f"BestatML[PersistenceError]: Model artifact does not exist.\n\n"
            f"Artifact:\n    {target}\n\nHow to fix:\n    Check the path and filename."
        )
    if not target.is_file():
        raise BestatMLPersistenceError(
            f"BestatML[PersistenceError]: Artifact path is not a regular file.\n\nArtifact:\n    {target}"
        )
    try:
        payload = joblib.load(target)
    except Exception as exc:
        raise BestatMLPersistenceError(
            f"BestatML[PersistenceError]: Could not load the model artifact.\n\n"
            f"Artifact:\n    {target}\n\nProblem:\n    {exc}\n\n"
            "The file may be corrupted, created by an incompatible environment, or untrusted. "
            "Only load artifacts you trust."
        ) from exc
    if not isinstance(payload, dict) or "model" not in payload:
        raise BestatMLPersistenceError(
            "BestatML[PersistenceError]: Invalid artifact format.\n\n"
            "Expected a BestatML artifact containing model metadata and a model payload."
        )
    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise BestatMLPersistenceError(
            "BestatML[PersistenceError]: Unsupported artifact format.\n\n"
            f"Detected format:\n    {payload.get('format_version')!r}\n"
            f"Supported format:\n    {ARTIFACT_FORMAT_VERSION}\n\n"
            "How to fix:\n    Load the artifact with a compatible BestatML version."
        )
    model = payload["model"]
    if not isinstance(model, cls):
        raise BestatMLPersistenceError(
            "BestatML[PersistenceError]: The artifact contains a different estimator type.\n\n"
            f"Stored estimator:\n    {type(model).__name__}\n"
            f"Requested loader:\n    {cls.__name__}.load()\n\n"
            "How to fix:\n    Use the matching BestatClassifier.load() or BestatRegressor.load()."
        )
    if not isinstance(payload.get("library_version"), str):
        raise BestatMLPersistenceError(
            "BestatML[PersistenceError]: Artifact is missing library-version metadata."
        )
    return model
