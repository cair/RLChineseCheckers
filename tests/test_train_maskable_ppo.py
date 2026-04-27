import unittest

from rl_agent.train_maskable_ppo import (
    checkpoint_stem,
    infer_episode_from_checkpoint,
    opponent_for_stage,
    player_count_for_stage,
)


class TrainMaskablePPOTests(unittest.TestCase):
    def test_checkpoint_stem_uses_model_stage_and_episode(self):
        self.assertEqual(
            checkpoint_stem("maskable_ppo", "empty_board", 1000),
            "maskable_ppo_empty_board_ep1000",
        )

    def test_infer_episode_from_checkpoint(self):
        self.assertEqual(infer_episode_from_checkpoint("checkpoints/maskable_ppo_self_play_ep4200.zip"), 4200)
        self.assertEqual(infer_episode_from_checkpoint(None), 0)

    def test_curriculum_stage_player_count_and_opponent(self):
        self.assertEqual(player_count_for_stage("empty_board", 6), 1)
        self.assertEqual(player_count_for_stage("self_play", 3), 3)
        self.assertEqual(opponent_for_stage("empty_board"), "none")
        self.assertEqual(opponent_for_stage("self_play"), "heuristic")


if __name__ == "__main__":
    unittest.main()
