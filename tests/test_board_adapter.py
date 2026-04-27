import unittest

from rl_agent import board_adapter
from rl_agent.config import COLOUR_ORDER


class BoardAdapterTests(unittest.TestCase):
    def test_metadata_matches_provided_board(self):
        meta = board_adapter.metadata()

        self.assertEqual(meta.board_size, 121)
        self.assertEqual(set(meta.colours), set(COLOUR_ORDER))
        for colour in COLOUR_ORDER:
            self.assertEqual(len(meta.start_indices[colour]), 10)
            self.assertEqual(len(meta.goal_indices[colour]), 10)
            self.assertEqual(meta.opposites[meta.opposites[colour]], colour)

    def test_initial_state_allows_one_player_empty_board_curriculum(self):
        state = board_adapter.initial_state(player_count=1)
        self.assertEqual(state.player_colours, ("red",))
        self.assertEqual(len(state.pins_for("red")), 10)

    def test_initial_state_can_start_from_any_controlled_colour(self):
        state = board_adapter.initial_state(player_count=2, controlled_colour="lawn green")
        self.assertEqual(state.current_colour, "lawn green")
        self.assertEqual(state.player_colours, ("lawn green", "gray0"))

    def test_legal_moves_match_provided_pin_rules_initial_state(self):
        state = board_adapter.initial_state(player_count=2)
        board, pins_by_colour_obj = board_adapter._build_board_and_pins(state.pins_by_colour)
        provided = {
            (pin.id, dest)
            for pin in pins_by_colour_obj[state.current_colour]
            for dest in pin.getPossibleMoves()
        }
        adapted = {
            (move.pin_id, move.to_index)
            for move in board_adapter.legal_moves(state)
        }

        self.assertEqual(adapted, provided)
        self.assertGreater(len(adapted), 0)


if __name__ == "__main__":
    unittest.main()
