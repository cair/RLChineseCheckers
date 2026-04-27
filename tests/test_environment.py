import unittest

from rl_agent.config import COLOUR_ORDER, DEFAULT_MAX_MOVES, EnvConfig, RANDOM_CONTROLLED_COLOUR
from rl_agent.environment import ChineseCheckersEnv


class EnvironmentTests(unittest.TestCase):
    def test_reset_supports_two_and_six_players(self):
        env = ChineseCheckersEnv(EnvConfig(player_count=2))
        self.assertEqual(len(env.state.player_colours), 2)
        self.assertGreater(len(env.legal_moves()), 0)

        state = env.reset(player_count=6)
        self.assertEqual(len(state.player_colours), 6)
        self.assertGreater(len(env.legal_moves()), 0)

    def test_step_accepts_legal_move_and_advances_turn(self):
        env = ChineseCheckersEnv(EnvConfig(player_count=2))
        first_colour = env.state.current_colour
        move = env.legal_moves()[0]
        result = env.step(move)

        self.assertEqual(result.state.move_count, 1)
        self.assertNotEqual(result.state.current_colour, first_colour)
        self.assertIn("reward_breakdown", result.info)
        self.assertIn("score", result.info)

    def test_score_helpers_return_numbers(self):
        env = ChineseCheckersEnv(EnvConfig(player_count=2))
        colour = env.state.current_colour
        self.assertIsInstance(env.pins_in_goal(env.state, colour), int)
        self.assertIsInstance(env.total_distance(env.state, colour), int)
        self.assertIsInstance(env.score_for_colour(env.state, colour), float)

    def test_default_max_moves_is_agent_move_cap(self):
        env = ChineseCheckersEnv(EnvConfig(player_count=3))
        self.assertEqual(env.config.effective_max_moves, DEFAULT_MAX_MOVES)

    def test_random_controlled_colour_uses_valid_colour(self):
        env = ChineseCheckersEnv(EnvConfig(player_count=2, controlled_colour=RANDOM_CONTROLLED_COLOUR, seed=1))
        self.assertIn(env.controlled_colour, COLOUR_ORDER)
        self.assertEqual(env.state.current_colour, env.controlled_colour)

    def test_max_moves_counts_per_colour(self):
        env = ChineseCheckersEnv(EnvConfig(player_count=2, max_moves=1))
        result = env.step(env.legal_moves()[0])
        self.assertTrue(result.truncated)
        self.assertEqual(result.info["active_colour_move_count"], 1)

    def test_goal_entry_reward_is_once_per_pin_per_episode(self):
        env = ChineseCheckersEnv(EnvConfig(player_count=2))
        colour = env.state.current_colour
        pin_id = 0
        goal_index = next(iter(__import__("rl_agent.board_adapter").board_adapter.metadata().goal_indices[colour]))
        env.state = env.state.__class__(
            pins_by_colour={
                **env.state.pins_by_colour,
                colour: tuple(goal_index if i == pin_id else idx for i, idx in enumerate(env.state.pins_for(colour))),
            },
            player_colours=env.state.player_colours,
            turn_order=env.state.turn_order,
            current_turn_index=env.state.current_turn_index,
            move_count=env.state.move_count,
            status=env.state.status,
            last_move=env.state.last_move,
        )
        env._goal_entry_rewarded_pins[colour].add(pin_id)
        self.assertIn(pin_id, env._goal_entry_rewarded_pins[colour])
        env.reset()
        self.assertEqual(env._goal_entry_rewarded_pins[env.controlled_colour], set())


if __name__ == "__main__":
    unittest.main()
