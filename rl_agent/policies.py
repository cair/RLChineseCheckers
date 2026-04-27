"""Baseline policies used for fallback, evaluation, and training scaffolds."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from .state import GameState, LegalMove


class Policy(Protocol):
    name: str

    def select_move(self, state: GameState, legal_moves: Sequence[LegalMove]) -> LegalMove:
        ...


@dataclass
class RandomPolicy:
    seed: int = 460
    name: str = "random"

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def select_move(self, state: GameState, legal_moves: Sequence[LegalMove]) -> LegalMove:
        if not legal_moves:
            raise ValueError("No legal moves available")
        return self.rng.choice(list(legal_moves))


@dataclass
class GreedyPolicy:
    name: str = "greedy"

    def select_move(self, state: GameState, legal_moves: Sequence[LegalMove]) -> LegalMove:
        if not legal_moves:
            raise ValueError("No legal moves available")
        return max(
            legal_moves,
            key=lambda move: (
                move.distance_delta,
                move.enters_goal,
                move.is_jump,
                -move.pin_id,
            ),
        )


@dataclass
class HeuristicPolicy:
    """Stronger deterministic policy that balances progress and setup."""

    name: str = "heuristic"

    def select_move(self, state: GameState, legal_moves: Sequence[LegalMove]) -> LegalMove:
        if not legal_moves:
            raise ValueError("No legal moves available")
        return max(legal_moves, key=self.score_move)

    def score_move(self, move: LegalMove) -> float:
        score = 0.0
        score += 5.0 * move.distance_delta
        score += 25.0 if move.enters_goal else 0.0
        score += 2.0 if move.is_jump and move.distance_delta >= 0 else 0.0
        score += min(move.future_jump_options, 6) * 1.5
        if move.distance_delta < 0:
            score -= 8.0 * abs(move.distance_delta)
        # Prefer moving lagging/high-id pieces only as a deterministic tie breaker.
        score -= move.pin_id * 0.01
        return score


def make_policy(name: str, seed: int = 460) -> Policy:
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "greedy":
        return GreedyPolicy()
    if name == "heuristic":
        return HeuristicPolicy()
    raise ValueError(f"Unknown policy: {name}")
