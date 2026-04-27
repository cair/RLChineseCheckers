"""Feature extraction helpers kept separate for report clarity."""

from __future__ import annotations

from . import board_adapter
from .config import COLOUR_ORDER
from .state import GameState, LegalMove


def state_features(state: GameState) -> list[float]:
    """Return a compact colour-relative numeric state summary.

    The active player is always encoded in the first colour slot. Other players
    follow turn order, then absent colours fill the remaining fixed slots. This
    lets one policy handle red, blue, green, yellow, gray, and purple without
    learning a separate meaning for each absolute colour name.
    """

    features: list[float] = [
        float(state.move_count),
        float(len(state.player_colours)),
        0.0,
    ]
    for colour in _colour_slots(state):
        pins = state.pins_by_colour.get(colour, ())
        if not pins:
            features.extend([0.0, 0.0, 0.0])
            continue
        distances = [board_adapter.min_distance_to_goal(idx, colour) for idx in pins]
        pins_in_goal = sum(idx in board_adapter.metadata().goal_indices[colour] for idx in pins)
        features.extend([
            float(pins_in_goal),
            float(sum(distances)),
            float(min(distances) if distances else 0),
        ])
    return features


def _colour_slots(state: GameState) -> tuple[str, ...]:
    active = state.current_colour
    ordered = [active]
    ordered.extend(colour for colour in state.turn_order if colour != active)
    ordered.extend(colour for colour in COLOUR_ORDER if colour not in ordered)
    return tuple(ordered[: len(COLOUR_ORDER)])


def move_features(state: GameState, move: LegalMove) -> list[float]:
    """Return simple per-move features for heuristics and custom rankers."""

    colour = state.current_colour
    pins = state.pins_for(colour)
    pins_in_goal = sum(idx in board_adapter.metadata().goal_indices[colour] for idx in pins)
    return [
        float(move.pin_id),
        float(move.from_index),
        float(move.to_index),
        float(move.distance_delta),
        1.0 if move.enters_goal else 0.0,
        1.0 if move.is_step else 0.0,
        1.0 if move.is_jump else 0.0,
        float(move.future_jump_options),
        float(pins_in_goal),
    ]
