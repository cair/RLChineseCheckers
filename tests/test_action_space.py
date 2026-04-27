import unittest

from rl_agent import action_space
from rl_agent.environment import ChineseCheckersEnv


class ActionSpaceTests(unittest.TestCase):
    def test_action_mapping_round_trip(self):
        action = action_space.action_id(3, 42)
        decoded = action_space.decode_action(action)

        self.assertEqual(decoded.pin_id, 3)
        self.assertEqual(decoded.to_index, 42)
        self.assertEqual(decoded.action_id, action)

    def test_legal_action_mask_matches_legal_moves(self):
        env = ChineseCheckersEnv()
        legal = env.legal_moves()
        mask = action_space.legal_action_mask(legal)

        self.assertEqual(len(mask), action_space.ACTION_SPACE_SIZE)
        self.assertEqual(sum(mask), len({action_space.move_to_action(m) for m in legal}))
        self.assertTrue(any(mask))

    def test_invalid_action_ids_raise(self):
        with self.assertRaises(ValueError):
            action_space.decode_action(-1)
        with self.assertRaises(ValueError):
            action_space.decode_action(action_space.ACTION_SPACE_SIZE)


if __name__ == "__main__":
    unittest.main()
