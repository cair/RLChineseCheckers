import unittest

from rl_agent.environment import ChineseCheckersEnv
from rl_agent.policies import GreedyPolicy, HeuristicPolicy, RandomPolicy


class PolicyTests(unittest.TestCase):
    def test_baseline_policies_choose_legal_moves(self):
        env = ChineseCheckersEnv()
        legal = env.legal_moves()
        legal_pairs = {(m.pin_id, m.to_index) for m in legal}

        for policy in (RandomPolicy(seed=1), GreedyPolicy(), HeuristicPolicy()):
            move = policy.select_move(env.state, legal)
            self.assertIn((move.pin_id, move.to_index), legal_pairs)


if __name__ == "__main__":
    unittest.main()
