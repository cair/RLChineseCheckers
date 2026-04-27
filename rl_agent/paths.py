"""Path helpers for repository files and generated artifacts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROVIDED_MULTI_DIR = REPO_ROOT / "multi system single machine minimal"
PROVIDED_SINGLE_DIR = REPO_ROOT / "single system"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
METRICS_DIR = REPO_ROOT / "metrics"
SELECTED_CANDIDATE_PATH = CHECKPOINT_DIR / "selected_candidate.json"


def ensure_runtime_dirs() -> None:
    """Create directories used for generated checkpoints and metrics."""

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(exist_ok=True)


def repo_path(*parts: str) -> Path:
    """Return a path under the repository root."""

    return REPO_ROOT.joinpath(*parts)


def provided_multi_path(*parts: str) -> Path:
    """Return a path inside the provided multi-system implementation."""

    return PROVIDED_MULTI_DIR.joinpath(*parts)
