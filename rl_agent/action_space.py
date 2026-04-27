"""Fixed discrete action mapping and legal masks for RL libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from . import board_adapter
from .config import PINS_PER_PLAYER
from .state import LegalMove


BOARD_SIZE = board_adapter.metadata().board_size
ACTION_SPACE_SIZE = PINS_PER_PLAYER * BOARD_SIZE


@dataclass(frozen=True)
class ActionMapping:
    action_id: int
    pin_id: int
    to_index: int


def action_id(pin_id: int, to_index: int) -> int:
    if pin_id < 0 or pin_id >= PINS_PER_PLAYER:
        raise ValueError("pin_id outside action space")
    if to_index < 0 or to_index >= BOARD_SIZE:
        raise ValueError("to_index outside action space")
    return pin_id * BOARD_SIZE + to_index


def decode_action(action: int) -> ActionMapping:
    if action < 0 or action >= ACTION_SPACE_SIZE:
        raise ValueError("action outside action space")
    return ActionMapping(action_id=action, pin_id=action // BOARD_SIZE, to_index=action % BOARD_SIZE)


def move_to_action(move: LegalMove) -> int:
    return action_id(move.pin_id, move.to_index)


def action_to_legal_move(action: int, legal_moves: Sequence[LegalMove]) -> LegalMove | None:
    decoded = decode_action(int(action))
    for move in legal_moves:
        if move.pin_id == decoded.pin_id and move.to_index == decoded.to_index:
            return move
    return None


def legal_action_mask(legal_moves: Iterable[LegalMove]) -> list[bool]:
    mask = [False] * ACTION_SPACE_SIZE
    for move in legal_moves:
        mask[move_to_action(move)] = True
    return mask


def legal_action_ids(legal_moves: Iterable[LegalMove]) -> list[int]:
    return [move_to_action(move) for move in legal_moves]
