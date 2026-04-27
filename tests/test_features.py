import unittest

from rl_agent import board_adapter
from rl_agent.features import state_features


class FeatureTests(unittest.TestCase):
    def test_state_features_are_colour_relative_for_symmetric_starts(self):
        red_state = board_adapter.initial_state(player_count=2, controlled_colour="red")
        green_state = board_adapter.initial_state(player_count=2, controlled_colour="lawn green")

        self.assertEqual(state_features(red_state), state_features(green_state))

    def test_active_player_is_encoded_in_first_colour_slot(self):
        state = board_adapter.initial_state(player_count=2, controlled_colour="yellow")
        features = state_features(state)

        active_pins_in_goal = features[3]
        active_total_distance = features[4]
        self.assertEqual(active_pins_in_goal, 0.0)
        self.assertGreater(active_total_distance, 0.0)


if __name__ == "__main__":
    unittest.main()
