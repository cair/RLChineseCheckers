import json
import tempfile
import unittest
from pathlib import Path

from rl_agent.evaluate import normalize_candidate_paths


class EvaluateCandidateTests(unittest.TestCase):
    def test_normalize_skips_selected_and_resolves_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            custom = root / "custom.json"
            custom.write_text(json.dumps({"weights": [1, 2, 3]}), encoding="utf-8")
            zip_path = root / "ppo.zip"
            zip_path.write_text("placeholder", encoding="utf-8")
            sidecar = root / "ppo.json"
            sidecar.write_text(
                json.dumps({"source": "maskable_ppo", "checkpoint_path": str(zip_path)}),
                encoding="utf-8",
            )
            selected = root / "selected_candidate.json"
            selected.write_text(json.dumps({"checkpoint_path": str(custom)}), encoding="utf-8")

            normalized = normalize_candidate_paths([str(custom), str(sidecar), str(zip_path), str(selected)])

            self.assertEqual(normalized, [str(custom), str(zip_path)])


if __name__ == "__main__":
    unittest.main()
