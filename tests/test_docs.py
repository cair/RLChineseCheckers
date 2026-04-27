import unittest
from pathlib import Path


class DocsTests(unittest.TestCase):
    def test_readme_mentions_report_concepts(self):
        text = Path("rl_agent/README.md").read_text(encoding="utf-8")
        self.assertIn("Reward Components", text)
        self.assertIn("Action Masks", text)
        self.assertIn("tournament_player.py", text)


if __name__ == "__main__":
    unittest.main()
