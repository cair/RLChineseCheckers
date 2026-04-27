"""Simple custom move-ranking agent used as a dependency-free fallback."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .features import move_features
from .paths import CHECKPOINT_DIR, ensure_runtime_dirs
from .policies import HeuristicPolicy
from .state import GameState, LegalMove


DEFAULT_WEIGHTS = [0.0, 0.0, 0.0, 5.0, 25.0, 0.2, 2.0, 1.5, 0.1]


@dataclass
class MoveRanker:
    """Tiny linear move scorer that is easy to inspect in a report."""

    weights: list[float] = field(default_factory=lambda: list(DEFAULT_WEIGHTS))
    name: str = "custom_ranker"

    def score(self, state: GameState, move: LegalMove) -> float:
        return sum(w * x for w, x in zip(self.weights, move_features(state, move)))

    def select_move(self, state: GameState, legal_moves: Sequence[LegalMove]) -> LegalMove:
        if not legal_moves:
            raise ValueError("No legal moves available")
        return max(legal_moves, key=lambda move: self.score(state, move))

    def update_toward(self, state: GameState, chosen: LegalMove, target: LegalMove, lr: float = 0.0005) -> None:
        """Move weights so the target action scores higher than the chosen one."""

        chosen_features = move_features(state, chosen)
        target_features = move_features(state, target)
        for i, (target_value, chosen_value) in enumerate(zip(target_features, chosen_features)):
            self.weights[i] += lr * (target_value - chosen_value)

    def save(self, path: Path, parent_candidate_id: str | None = None, episodes: int = 0) -> Path:
        ensure_runtime_dirs()
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate_id = path.stem
        data = {
            "candidate_id": candidate_id,
            "source": "custom_move_ranker",
            "checkpoint_path": str(path),
            "parent_candidate_id": parent_candidate_id,
            "training_profile": "custom",
            "training_method": "custom_ranker",
            "episodes_completed": episodes,
            "created_at": time.time(),
            "weights": self.weights,
            "summary_metrics": {},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MoveRanker":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(weights=[float(v) for v in data["weights"]])


def new_candidate_path(prefix: str = "custom_ranker") -> Path:
    ensure_runtime_dirs()
    return CHECKPOINT_DIR / f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
