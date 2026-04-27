import tempfile
import unittest
from pathlib import Path

from rl_agent.agent import MoveRanker
from rl_agent.environment import ChineseCheckersEnv


class AgentTests(unittest.TestCase):
    def test_ranker_selects_legal_move_and_round_trips_checkpoint(self):
        env = ChineseCheckersEnv()
        legal = env.legal_moves()
        ranker = MoveRanker()
        move = ranker.select_move(env.state, legal)
        self.assertIn((move.pin_id, move.to_index), {(m.pin_id, m.to_index) for m in legal})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranker.json"
            ranker.save(path, episodes=1)
            loaded = MoveRanker.load(path)
            self.assertEqual(loaded.weights, ranker.weights)


if __name__ == "__main__":
    unittest.main()
