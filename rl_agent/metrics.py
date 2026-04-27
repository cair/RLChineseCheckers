"""CSV/JSON metric recording for training, evaluation, and reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .paths import METRICS_DIR, ensure_runtime_dirs


@dataclass
class EpisodeMetric:
    session_id: str
    episode: int
    candidate_id: str
    player_count: int
    opponent_profile: str
    training_method: str
    total_reward: float
    final_score: float
    moves: int
    pins_in_goal: int
    total_distance: int
    distance_progress: float
    avg_time_per_move_sec: float
    game_duration_sec: float
    won: bool
    draw: bool
    legal_move_rate: float = 1.0
    mask_legal_count: int = 0
    invalid_action_attempts: int = 0
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationMetric:
    run_id: str
    candidate_id: str
    candidate_source: str
    opponent_profile: str
    player_count: int
    games: int
    avg_reward: float
    avg_score: float
    avg_moves: float
    avg_pins_in_goal: float
    avg_total_distance: float
    avg_distance_progress: float
    avg_time_per_move_sec: float
    avg_game_duration_sec: float
    legal_move_rate: float
    action_mask_match_rate: float = 1.0
    selected_candidate: bool = False


def flatten_metric(metric) -> dict:
    data = asdict(metric)
    reward_components = data.pop("reward_components", None)
    if reward_components:
        for key, value in reward_components.items():
            data[f"reward_{key}"] = value
    return data


def write_jsonl(path: Path, rows: Iterable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(flatten_metric(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable) -> None:
    rows = [flatten_metric(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_metrics_path(name: str, suffix: str = "jsonl") -> Path:
    ensure_runtime_dirs()
    return METRICS_DIR / f"{name}.{suffix}"


def summarize_metrics(metrics_dir: Path) -> dict[str, dict[str, float]]:
    """Return simple report-friendly means for numeric JSONL metrics."""

    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for path in metrics_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("candidate_id", path.stem)
            totals.setdefault(name, {})
            counts[name] = counts.get(name, 0) + 1
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[name][key] = totals[name].get(key, 0.0) + float(value)
    return {
        name: {key: value / counts[name] for key, value in values.items()}
        for name, values in totals.items()
        if counts.get(name)
    }
