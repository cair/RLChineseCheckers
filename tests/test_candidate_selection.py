import unittest

from rl_agent.evaluate import EvaluationMetric, select_best_candidate


class CandidateSelectionTests(unittest.TestCase):
    def test_select_best_candidate_prefers_score_then_progress(self):
        weak = EvaluationMetric("r", "weak", "heuristic", "random", 2, 1, 0, 10, 10, 0, 50, 1, 0, 0, 1)
        strong = EvaluationMetric("r", "strong", "custom", "random", 2, 1, 0, 20, 12, 0, 40, 2, 0, 0, 1)

        self.assertEqual(select_best_candidate([weak, strong]).candidate_id, "strong")


if __name__ == "__main__":
    unittest.main()
