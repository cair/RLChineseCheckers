"""Adapter around the provided Chinese Checkers board and pin rules.

The course files own the authoritative board geometry and legal move rules.
This module keeps those mutable classes behind a small project-owned interface
so training, evaluation, and tests can use immutable snapshots.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .config import COLOUR_ORDER, OPPOSITE_COLOURS, PINS_PER_PLAYER
from .errors import InvalidStateError
from .paths import PROVIDED_MULTI_DIR
from .state import GameState, LegalMove, copy_pins


def _ensure_provided_path() -> None:
    provided = str(PROVIDED_MULTI_DIR)
    if provided not in sys.path:
        sys.path.insert(0, provided)


def load_provided_classes():
    """Return the provided `HexBoard` and `Pin` classes."""

    _ensure_provided_path()
    board_mod = importlib.import_module("checkers_board")
    pins_mod = importlib.import_module("checkers_pins")
    return board_mod.HexBoard, pins_mod.Pin


def new_board():
    """Create a fresh provided board instance."""

    HexBoard, _ = load_provided_classes()
    return HexBoard()


@dataclass(frozen=True)
class BoardMetadata:
    board_size: int
    colours: Tuple[str, ...]
    opposites: Mapping[str, str]
    start_indices: Mapping[str, Tuple[int, ...]]
    goal_indices: Mapping[str, Tuple[int, ...]]
    axial_by_index: Mapping[int, Tuple[int, int]]


@lru_cache(maxsize=1)
def metadata() -> BoardMetadata:
    """Compute stable board metadata from the provided board."""

    board = new_board()
    start_indices = {
        colour: tuple(board.axial_of_colour(colour)[:PINS_PER_PLAYER])
        for colour in COLOUR_ORDER
    }
    goal_indices = {
        colour: tuple(board.axial_of_colour(OPPOSITE_COLOURS[colour])[:PINS_PER_PLAYER])
        for colour in COLOUR_ORDER
    }
    axial_by_index = {
        idx: (cell.q, cell.r)
        for idx, cell in enumerate(board.cells)
    }
    return BoardMetadata(
        board_size=len(board.cells),
        colours=COLOUR_ORDER,
        opposites=OPPOSITE_COLOURS,
        start_indices=start_indices,
        goal_indices=goal_indices,
        axial_by_index=axial_by_index,
    )


def axial_distance(a_index: int, b_index: int) -> int:
    """Hex distance between two board indices."""

    meta = metadata()
    aq, ar = meta.axial_by_index[a_index]
    bq, br = meta.axial_by_index[b_index]
    a_s = -aq - ar
    b_s = -bq - br
    return max(abs(aq - bq), abs(ar - br), abs(a_s - b_s))


def min_distance_to_goal(index: int, colour: str) -> int:
    """Minimum hex distance from a cell to the colour's target zone."""

    return min(axial_distance(index, goal) for goal in metadata().goal_indices[colour])


def default_colours(player_count: int) -> Tuple[str, ...]:
    """Deterministic local colour assignment for 1-6 local training players."""

    if player_count < 1 or player_count > len(COLOUR_ORDER):
        raise ValueError("player_count must be between 1 and 6")
    return COLOUR_ORDER[:player_count]


def colours_for_controlled_player(player_count: int, controlled_colour: str = "red") -> Tuple[str, ...]:
    """Return local training colours with the controlled colour moving first."""

    if player_count < 1 or player_count > len(COLOUR_ORDER):
        raise ValueError("player_count must be between 1 and 6")
    if controlled_colour not in COLOUR_ORDER:
        raise ValueError(f"Unknown controlled colour: {controlled_colour}")

    colours = [controlled_colour]
    opposite = OPPOSITE_COLOURS[controlled_colour]
    if player_count > 1:
        colours.append(opposite)
    for colour in COLOUR_ORDER:
        if len(colours) >= player_count:
            break
        if colour not in colours:
            colours.append(colour)
    return tuple(colours)


def initial_pins(player_colours: Sequence[str]) -> Dict[str, Tuple[int, ...]]:
    """Return starting pin indices for the given colours."""

    meta = metadata()
    return {colour: meta.start_indices[colour] for colour in player_colours}


def initial_state(player_count: int = 2, controlled_colour: str = "red") -> GameState:
    """Create a deterministic initial local game state."""

    ordered = colours_for_controlled_player(player_count, controlled_colour)
    pins = initial_pins(ordered)
    return GameState(pins_by_colour=pins, player_colours=ordered, turn_order=ordered)


def _build_board_and_pins(pins_by_colour: Mapping[str, Sequence[int]]):
    """Create provided board/pin objects for a snapshot."""

    board = new_board()
    _, Pin = load_provided_classes()
    all_positions = [idx for pins in pins_by_colour.values() for idx in pins]
    if len(all_positions) != len(set(all_positions)):
        raise InvalidStateError("Two pins occupy the same board cell")
    if any(idx < 0 or idx >= len(board.cells) for idx in all_positions):
        raise InvalidStateError("Pin index outside board")

    pins_by_colour_obj = {}
    for colour, indices in pins_by_colour.items():
        pins_by_colour_obj[colour] = [
            Pin(board, int(index), id=pin_id, color=colour)
            for pin_id, index in enumerate(indices)
        ]
    return board, pins_by_colour_obj


def legal_moves_for_colour(state: GameState, colour: str) -> Tuple[LegalMove, ...]:
    """Return all legal moves for one colour using the provided rules."""

    if colour not in state.pins_by_colour:
        raise InvalidStateError(f"Unknown colour: {colour}")
    _, pins_by_colour_obj = _build_board_and_pins(state.pins_by_colour)
    moves: List[LegalMove] = []
    for pin in pins_by_colour_obj[colour]:
        from_index = int(pin.axialindex)
        before_goal_dist = min_distance_to_goal(from_index, colour)
        for to_index in pin.getPossibleMoves():
            dist = axial_distance(from_index, int(to_index))
            after_goal_dist = min_distance_to_goal(int(to_index), colour)
            moves.append(
                LegalMove(
                    pin_id=int(pin.id),
                    from_index=from_index,
                    to_index=int(to_index),
                    is_step=dist == 1,
                    is_jump=dist > 1,
                    distance_delta=before_goal_dist - after_goal_dist,
                    enters_goal=int(to_index) in metadata().goal_indices[colour],
                )
            )
    return tuple(sorted(moves, key=lambda m: (m.pin_id, m.to_index)))


def legal_moves(state: GameState) -> Tuple[LegalMove, ...]:
    """Return legal moves for the active colour."""

    return legal_moves_for_colour(state, state.current_colour)


def apply_move_to_pins(
    pins_by_colour: Mapping[str, Sequence[int]],
    colour: str,
    move: LegalMove,
) -> Dict[str, Tuple[int, ...]]:
    """Return a new pin mapping with the given move applied."""

    pins = copy_pins(pins_by_colour)
    colour_pins = list(pins[colour])
    if colour_pins[move.pin_id] != move.from_index:
        raise InvalidStateError("Move source does not match state")
    occupied = {idx for c, values in pins.items() for idx in values if c != colour}
    occupied.update(idx for i, idx in enumerate(colour_pins) if i != move.pin_id)
    if move.to_index in occupied:
        raise InvalidStateError("Move destination is occupied")
    colour_pins[move.pin_id] = move.to_index
    pins[colour] = tuple(colour_pins)
    return pins
