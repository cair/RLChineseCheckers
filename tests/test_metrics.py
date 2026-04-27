import tempfile
import unittest
from pathlib import Path

from rl_agent.metrics import EvaluationMetric, EpisodeMetric, write_csv, write_jsonl


class MetricsTests(unittest.TestCase):
    def test_writers_include_required_fields(self):
        metric = EvaluationMetric(
            run_id="run",
            candidate_id="heuristic",
            candidate_source="heuristic",
            opponent_profile="random",
            player_count=2,
            games=1,
            avg_reward=1.0,
            avg_score=2.0,
            avg_moves=3.0,
            avg_pins_in_goal=0.0,
            avg_total_distance=10.0,
            avg_distance_progress=1.0,
            avg_time_per_move_sec=0.01,
            avg_game_duration_sec=0.1,
            legal_move_rate=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "metrics.jsonl"
            csv_path = Path(tmp) / "metrics.csv"
            write_jsonl(json_path, [metric])
            write_csv(csv_path, [metric])
            self.assertIn("avg_score", json_path.read_text(encoding="utf-8"))
            self.assertIn("avg_score", csv_path.read_text(encoding="utf-8"))

    def test_episode_metric_flattens_reward_components(self):
        metric = EpisodeMetric(
            session_id="s",
            episode=1,
            candidate_id="c",
            player_count=2,
            opponent_profile="random",
            training_method="heuristic",
            total_reward=1.0,
            final_score=1.0,
            moves=1,
            pins_in_goal=0,
            total_distance=10,
            distance_progress=1.0,
            avg_time_per_move_sec=0.1,
            game_duration_sec=0.1,
            won=False,
            draw=False,
            reward_components={"distance_progress": 1.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "episode.csv"
            write_csv(csv_path, [metric])
            self.assertIn("reward_distance_progress", csv_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
