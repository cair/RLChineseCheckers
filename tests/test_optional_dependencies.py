import unittest

from rl_agent.errors import OptionalDependencyMissing
from rl_agent.train_maskable_ppo import require_maskable_ppo


class OptionalDependencyTests(unittest.TestCase):
    def test_maskable_ppo_dependency_message_or_import(self):
        try:
            deps = require_maskable_ppo()
        except OptionalDependencyMissing as exc:
            self.assertIn("MaskablePPO training requires", str(exc))
        else:
            self.assertEqual(len(deps), 3)


if __name__ == "__main__":
    unittest.main()
