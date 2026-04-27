import unittest

from rl_agent.tournament_player import choose_move, legal_moves_from_server, state_from_server


SERVER_STATE = {
    "pins": {
        "red": [111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
        "blue": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "turn_order": ["red", "blue"],
    "current_turn_colour": "red",
    "move_count": 0,
    "status": "PLAYING",
}


class TournamentAdapterTests(unittest.TestCase):
    def test_state_from_server(self):
        state = state_from_server(SERVER_STATE, "red")
        self.assertEqual(state.current_colour, "red")
        self.assertEqual(len(state.pins_for("red")), 10)

    def test_legal_moves_from_server_filters_shape(self):
        legal = {"0": [102, 103], "1": [104]}
        moves = legal_moves_from_server(SERVER_STATE, "red", legal)
        self.assertEqual({(m.pin_id, m.to_index) for m in moves}, {(0, 102), (0, 103), (1, 104)})

    def test_choose_move_uses_legal_move_only(self):
        legal = {"0": [102, 103], "1": [104]}
        move = choose_move(SERVER_STATE, "red", legal)
        self.assertIn((move.pin_id, move.to_index), {(0, 102), (0, 103), (1, 104)})


if __name__ == "__main__":
    unittest.main()
