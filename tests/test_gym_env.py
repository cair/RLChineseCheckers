import unittest

from rl_agent.action_space import action_to_legal_move, legal_action_mask
from rl_agent.errors import OptionalDependencyMissing
from rl_agent.gym_env import RLChineseCheckersGymEnv


class GymEnvTests(unittest.TestCase):
    def make_env(self):
        try:
            return RLChineseCheckersGymEnv()
        except OptionalDependencyMissing as exc:
            self.skipTest(str(exc))

    def test_gym_env_reset_and_mask(self):
        env = self.make_env()
        obs, info = env.reset()
        mask = env.action_masks()

        self.assertEqual(len(mask), env.action_space.n)
        self.assertGreater(sum(mask), 0)
        self.assertIn("legal_move_count", info)
        self.assertEqual(info["legal_move_count"], sum(mask))
        self.assertEqual(len(obs), env.observation_space.shape[0])

    def test_action_mask_matches_direct_legal_moves_after_steps(self):
        env = self.make_env()
        env.reset()
        for _ in range(3):
            legal = env.local_env.legal_moves()
            mask = env.action_masks()
            self.assertEqual(mask, legal_action_mask(legal))
            action = next(i for i, allowed in enumerate(mask) if allowed)
            self.assertIsNotNone(action_to_legal_move(action, legal))
            env.step(action)


if __name__ == "__main__":
    unittest.main()
