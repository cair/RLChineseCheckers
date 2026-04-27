"""Named reward components for explainable Chinese Checkers training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .state import LegalMove


@dataclass(frozen=True)
class RewardBreakdown:
    distance_progress: float = 0.0
    goal_entry: float = 0.0
    move_penalty: float = 0.0
    regression_penalty: float = 0.0
    jump_progress: float = 0.0
    future_jump_setup: float = 0.0
    finish_bonus: float = 0.0
    invalid_or_timeout_penalty: float = 0.0

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, float]:
        data = asdict(self)
        data["total"] = self.total
        return data


def reward_for_move(
    move: LegalMove,
    first_time_goal_entries: int = 0,
    pins_in_goal: int = 0,
    won: bool = False,
    episode_done: bool = False,
    invalid: bool = False,
) -> RewardBreakdown:
    """Compute shaped reward while paying goal entry once per pin per episode."""

    if invalid:
        return RewardBreakdown(invalid_or_timeout_penalty=-10.0)

    progress = float(move.distance_delta)
    return RewardBreakdown(
        distance_progress=progress * 2.0,
        goal_entry=float(first_time_goal_entries) * 10.0,
        move_penalty=-0.05,
        regression_penalty=-2.0 * abs(progress) if progress < 0 else 0.0,
        jump_progress=0.25 if move.is_jump and progress >= 0 else 0.0,
        future_jump_setup=min(float(move.future_jump_options), 4.0) * 0.25,
        finish_bonus=float(pins_in_goal) * 50.0 if episode_done or won else 0.0,
    )
