"""Optional Weights & Biases tracking helpers.

The rest of the project can call this module without depending on wandb being
installed. If wandb is missing or disabled, the tracker becomes a no-op.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


class WandbTracker:
    """Tiny wrapper around wandb with a no-op fallback."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        self.enabled = False
        self._wandb = None
        self.run = None
        if not enabled:
            return
        try:
            import wandb
        except Exception:
            print("wandb is not installed; continuing without W&B tracking.")
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=dict(config or {}),
            tags=tags,
        )
        self.enabled = True

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(dict(data), step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()


def flatten_for_wandb(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten dataclasses and nested dicts into wandb-friendly scalar keys."""

    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(flatten_for_wandb(full_key, item))
        else:
            flat[full_key] = item
    return flat
