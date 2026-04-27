"""Shared state dataclasses for the local agent environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class LegalMove:
    """One legal move for the active player."""

    pin_id: int
    from_index: int
    to_index: int
    is_step: bool
    is_jump: bool
    distance_delta: int = 0
    enters_goal: bool = False
    future_jump_options: int = 0


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a Chinese Checkers game."""

    pins_by_colour: Mapping[str, Tuple[int, ...]]
    player_colours: Tuple[str, ...]
    turn_order: Tuple[str, ...]
    current_turn_index: int = 0
    move_count: int = 0
    status: str = "PLAYING"
    last_move: LegalMove | None = None
    scores: Mapping[str, float] = field(default_factory=dict)

    @property
    def current_colour(self) -> str:
        return self.turn_order[self.current_turn_index]

    def pins_for(self, colour: str) -> Tuple[int, ...]:
        return tuple(self.pins_by_colour[colour])


def copy_pins(pins_by_colour: Mapping[str, Tuple[int, ...] | list[int]]) -> Dict[str, Tuple[int, ...]]:
    """Normalize pin containers to immutable tuples."""

    return {colour: tuple(indices) for colour, indices in pins_by_colour.items()}
