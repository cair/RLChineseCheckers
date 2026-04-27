import unittest

from rl_agent.rewards import RewardBreakdown, reward_for_move
from rl_agent.state import LegalMove


class RewardTests(unittest.TestCase):
    def test_reward_breakdown_total_matches_components(self):
        breakdown = RewardBreakdown(
            distance_progress=1.0,
            goal_entry=2.0,
            move_penalty=-0.05,
            regression_penalty=-0.5,
            jump_progress=0.5,
        )
        data = breakdown.to_dict()
        component_total = sum(v for k, v in data.items() if k != "total")
        self.assertAlmostEqual(data["total"], component_total)

    def test_reward_for_move_names_components(self):
        move = LegalMove(
            pin_id=0,
            from_index=10,
            to_index=20,
            is_step=False,
            is_jump=True,
            distance_delta=2,
            enters_goal=False,
            future_jump_options=3,
        )
        data = reward_for_move(move, first_time_goal_entries=1, pins_in_goal=1).to_dict()
        for key in (
            "distance_progress",
            "goal_entry",
            "move_penalty",
            "regression_penalty",
            "jump_progress",
            "future_jump_setup",
            "finish_bonus",
            "invalid_or_timeout_penalty",
            "total",
        ):
            self.assertIn(key, data)

    def test_reward_keeps_original_shaping_and_once_per_pin_goal_entry(self):
        move = LegalMove(
            pin_id=0,
            from_index=10,
            to_index=20,
            is_step=False,
            is_jump=True,
            distance_delta=4,
            enters_goal=True,
            future_jump_options=4,
        )
        data = reward_for_move(move, first_time_goal_entries=1, pins_in_goal=1).to_dict()
        self.assertAlmostEqual(data["distance_progress"], 8.0)
        self.assertAlmostEqual(data["jump_progress"], 0.25)
        self.assertAlmostEqual(data["future_jump_setup"], 1.0)
        self.assertAlmostEqual(data["goal_entry"], 10.0)
        self.assertAlmostEqual(data["move_penalty"], -0.05)

    def test_terminal_bonus_depends_on_current_pins_in_goal(self):
        move = LegalMove(
            pin_id=0,
            from_index=10,
            to_index=20,
            is_step=True,
            is_jump=False,
            distance_delta=0,
            enters_goal=False,
        )
        data = reward_for_move(move, pins_in_goal=3, episode_done=True).to_dict()
        self.assertAlmostEqual(data["finish_bonus"], 150.0)


if __name__ == "__main__":
    unittest.main()
