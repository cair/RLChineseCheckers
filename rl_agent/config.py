"""Small configuration objects used across the RL Chinese Checkers agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


COLOUR_ORDER: Tuple[str, ...] = ("red", "lawn green", "yellow", "blue", "gray0", "purple")
PRIMARY_COLOURS: Tuple[str, ...] = ("red", "lawn green", "yellow")
OPPOSITE_COLOURS = {
    "red": "blue",
    "blue": "red",
    "lawn green": "gray0",
    "gray0": "lawn green",
    "yellow": "purple",
    "purple": "yellow",
}

DEFAULT_SEED = 460
DEFAULT_WANDB_PROJECT = "rlcc"
RANDOM_CONTROLLED_COLOUR = "random"
MIN_PLAYERS = 2
MAX_PLAYERS = 6
PINS_PER_PLAYER = 10
TOURNAMENT_TARGET_MOVES_PER_PLAYER = 45
DEFAULT_MAX_MOVES = 100


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for local simulated games."""

    player_count: int = 2
    controlled_colour: str = "red"
    max_moves: int | None = None
    seed: int = DEFAULT_SEED

    @property
    def effective_max_moves(self) -> int:
        return self.max_moves if self.max_moves is not None else DEFAULT_MAX_MOVES


@dataclass(frozen=True)
class TrainingProfile:
    """A simple named training profile for practice or longer runs."""

    name: str
    episodes: int
    player_counts: Tuple[int, ...]
    opponent_profile: str
    description: str


PRACTICE_PROFILE = TrainingProfile(
    name="practice",
    episodes=100,
    player_counts=(2, 3, 4),
    opponent_profile="mixed_baseline",
    description="Short run intended for practice tournament preparation.",
)

EXTENDED_PROFILE = TrainingProfile(
    name="extended",
    episodes=1000,
    player_counts=(2, 3, 4, 5, 6),
    opponent_profile="mixed_baseline_selfplay",
    description="Longer run intended for continued improvement from a checkpoint.",
)

CURRICULUM_PROFILE = TrainingProfile(
    name="curriculum",
    episodes=500,
    player_counts=(2, 3, 4, 5, 6),
    opponent_profile="curriculum",
    description="Staged movement, baseline opponent, and mixed-player training.",
)

TRAINING_PROFILES = {
    PRACTICE_PROFILE.name: PRACTICE_PROFILE,
    EXTENDED_PROFILE.name: EXTENDED_PROFILE,
    CURRICULUM_PROFILE.name: CURRICULUM_PROFILE,
}

CURRICULUM_STAGES = (
    "movement_practice",
    "baseline_opponent_play",
    "mixed_player_counts",
    "self_play_snapshots",
)


@dataclass(frozen=True)
class CandidateInfo:
    """Metadata for saved policies and selected tournament candidates."""

    candidate_id: str
    source: str
    checkpoint_path: str
    parent_candidate_id: str | None = None
    summary_metrics: dict = field(default_factory=dict)
