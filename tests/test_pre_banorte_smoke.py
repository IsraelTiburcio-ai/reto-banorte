from __future__ import annotations

import contextlib
import io
import unittest

from scripts.pre_banorte_smoke import load_cases, run


class PreBanorteSmokeTests(unittest.TestCase):
    def test_fixture_contains_thirty_unique_cases(self) -> None:
        cases = load_cases()
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({case["id"] for case in cases}), 30)
        self.assertTrue(any(case["multi_turn"] for case in cases))
        self.assertTrue(any(case["stream"] for case in cases))

    def test_default_mode_is_offline_and_makes_no_http_calls(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = run(None, None)

        self.assertEqual(result, 0)
        self.assertIn("30/30", output.getvalue())
        self.assertIn("HTTP calls: 0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
