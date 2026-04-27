import unittest

from rl_agent.wandb_utils import WandbTracker, flatten_for_wandb


class WandbUtilsTests(unittest.TestCase):
    def test_disabled_tracker_is_noop(self):
        tracker = WandbTracker(enabled=False, project="rlcc")
        tracker.log({"x": 1}, step=1)
        tracker.finish()
        self.assertFalse(tracker.enabled)

    def test_flatten_nested_dict(self):
        flat = flatten_for_wandb("reward", {"a": 1, "nested": {"b": 2}})
        self.assertEqual(flat["reward/a"], 1)
        self.assertEqual(flat["reward/nested/b"], 2)


if __name__ == "__main__":
    unittest.main()
