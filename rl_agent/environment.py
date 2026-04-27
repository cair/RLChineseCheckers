"""Local Chinese Checkers simulator for training and evaluation.

This environment mirrors the provided move rules by calling the board adapter,
but it avoids sockets so training can run quickly and repeatably.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import board_adapter
from .config import COLOUR_ORDER, EnvConfig, RANDOM_CONTROLLED_COLOUR
from .errors import IllegalActionError
from .rewards import RewardBreakdown, reward_for_move
from .state import GameState, LegalMove


@dataclass(frozen=True)
class StepResult:
    state: GameState
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class ChineseCheckersEnv:
    """Direct Python API for local Chinese Checkers games."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.rng = random.Random(self.config.seed)
        self.controlled_colour = self._choose_controlled_colour(self.config.controlled_colour)
        self.state = board_adapter.initial_state(
            player_count=self.config.player_count,
            controlled_colour=self.controlled_colour,
        )
        self._moves_by_colour = {colour: 0 for colour in self.state.player_colours}
        self._goal_entry_rewarded_pins = {colour: set() for colour in self.state.player_colours}

    def reset(
        self,
        player_count: int | None = None,
        controlled_colour: str | None = None,
        seed: int | None = None,
    ) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        count = player_count or self.config.player_count
        colour = self._choose_controlled_colour(controlled_colour or self.config.controlled_colour)
        self.controlled_colour = colour
        self.state = board_adapter.initial_state(count, colour)
        self._moves_by_colour = {colour: 0 for colour in self.state.player_colours}
        self._goal_entry_rewarded_pins = {colour: set() for colour in self.state.player_colours}
        return self.state

    def legal_moves(self, colour: str | None = None, include_future_jump_count: bool = False) -> tuple[LegalMove, ...]:
        if colour is None or colour == self.state.current_colour:
            moves = board_adapter.legal_moves(self.state)
        else:
            moves = board_adapter.legal_moves_for_colour(self.state, colour)
        if not include_future_jump_count:
            return moves
        return tuple(self._with_future_jump_count(self.state, colour or self.state.current_colour, m) for m in moves)

    def step(self, move: LegalMove | tuple[int, int]) -> StepResult:
        legal = self.legal_moves()
        selected = self._coerce_move(move, legal)
        if selected is None:
            breakdown = reward_for_move(LegalMove(-1, -1, -1, False, False), invalid=True)
            raise IllegalActionError(f"Illegal move: {move}; reward={breakdown.total}")

        active = self.state.current_colour
        pins_in_goal_before = self.pins_in_goal(self.state, active)
        new_pins = board_adapter.apply_move_to_pins(self.state.pins_by_colour, active, selected)
        self._moves_by_colour[active] = self._moves_by_colour.get(active, 0) + 1
        active_colour_moves = self._moves_by_colour[active]
        next_index = (self.state.current_turn_index + 1) % len(self.state.turn_order)
        move_count = self.state.move_count + 1
        temp_state = GameState(
            pins_by_colour=new_pins,
            player_colours=self.state.player_colours,
            turn_order=self.state.turn_order,
            current_turn_index=next_index,
            move_count=move_count,
            status="PLAYING",
            last_move=selected,
        )

        won = self.has_won(temp_state, active)
        draw = not won and all(len(board_adapter.legal_moves_for_colour(temp_state, c)) == 0 for c in temp_state.player_colours)
        truncated = active_colour_moves >= self.config.effective_max_moves
        status = "WIN" if won else "DRAW" if draw else "TRUNCATED" if truncated else "PLAYING"
        next_state = GameState(
            pins_by_colour=new_pins,
            player_colours=self.state.player_colours,
            turn_order=self.state.turn_order,
            current_turn_index=next_index,
            move_count=move_count,
            status=status,
            last_move=selected,
        )
        pins_in_goal = self.pins_in_goal(next_state, active)
        new_goal_pins = max(0, pins_in_goal - pins_in_goal_before)
        goals = set(board_adapter.metadata().goal_indices[active])
        rewarded_pins = self._goal_entry_rewarded_pins.setdefault(active, set())
        first_time_goal_entries = 0
        for pin_id, pin_index in enumerate(new_pins[active]):
            if pin_index in goals and pin_id not in rewarded_pins:
                rewarded_pins.add(pin_id)
                first_time_goal_entries += 1
        episode_done = won or draw or truncated
        breakdown = reward_for_move(
            selected,
            first_time_goal_entries=first_time_goal_entries,
            pins_in_goal=pins_in_goal,
            won=won,
            episode_done=episode_done,
        )
        self.state = next_state
        info = {
            "selected_move": selected,
            "reward_breakdown": breakdown.to_dict(),
            "score": self.score_for_colour(next_state, active),
            "pins_in_goal": pins_in_goal,
            "new_goal_pins": new_goal_pins,
            "first_time_goal_entries": first_time_goal_entries,
            "total_distance": self.total_distance(next_state, active),
            "move_count": move_count,
            "active_colour_move_count": active_colour_moves,
            "legal_move_count": len(legal),
            "max_moves": self.config.effective_max_moves,
        }
        return StepResult(next_state, breakdown.total, won or draw, truncated, info)

    def has_won(self, state: GameState, colour: str) -> bool:
        goals = set(board_adapter.metadata().goal_indices[colour])
        return all(idx in goals for idx in state.pins_for(colour))

    def pins_in_goal(self, state: GameState, colour: str) -> int:
        goals = set(board_adapter.metadata().goal_indices[colour])
        return sum(idx in goals for idx in state.pins_for(colour))

    def total_distance(self, state: GameState, colour: str) -> int:
        return sum(board_adapter.min_distance_to_goal(idx, colour) for idx in state.pins_for(colour)
                   if idx not in board_adapter.metadata().goal_indices[colour])

    def score_for_colour(self, state: GameState, colour: str) -> float:
        pins_goal = self.pins_in_goal(state, colour)
        distance_score = max(0.0, 200.0 - float(self.total_distance(state, colour)))
        return pins_goal * 100.0 + distance_score

    def _coerce_move(self, move: LegalMove | tuple[int, int], legal: Sequence[LegalMove]) -> LegalMove | None:
        if isinstance(move, LegalMove):
            key = (move.pin_id, move.to_index)
        else:
            key = (int(move[0]), int(move[1]))
        for candidate in legal:
            if (candidate.pin_id, candidate.to_index) == key:
                return candidate
        return None

    def _choose_controlled_colour(self, configured_colour: str) -> str:
        if configured_colour == RANDOM_CONTROLLED_COLOUR:
            return self.rng.choice(COLOUR_ORDER)
        return configured_colour

    def _with_future_jump_count(self, state: GameState, colour: str, move: LegalMove) -> LegalMove:
        new_pins = board_adapter.apply_move_to_pins(state.pins_by_colour, colour, move)
        moved_state = GameState(
            pins_by_colour=new_pins,
            player_colours=state.player_colours,
            turn_order=state.turn_order,
            current_turn_index=state.current_turn_index,
            move_count=state.move_count,
            status=state.status,
            last_move=move,
        )
        future = board_adapter.legal_moves_for_colour(moved_state, colour)
        future_jump_options = sum(1 for m in future if m.is_jump)
        return LegalMove(
            pin_id=move.pin_id,
            from_index=move.from_index,
            to_index=move.to_index,
            is_step=move.is_step,
            is_jump=move.is_jump,
            distance_delta=move.distance_delta,
            enters_goal=move.enters_goal,
            future_jump_options=future_jump_options,
        )
