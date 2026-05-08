import contextlib
import io
import unittest

from pymegdec import cli_consolidated as cli


class TestCliConsolidation(unittest.TestCase):
    def test_top_level_help_returns_zero(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli.main([])

        self.assertEqual(status, 0)
        self.assertIn("stimulus predictions", output.getvalue())
        self.assertIn("alpha reaction-time", output.getvalue())

    def test_group_help_returns_zero(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli.main(["stimulus", "--help"])

        self.assertEqual(status, 0)
        self.assertIn("temporal-generalization", output.getvalue())
        self.assertIn("onset-scan", output.getvalue())

    def test_single_token_command_resolution(self):
        resolved = cli.resolve_command(["stimulus-predictions", "--output", "predictions.csv"])

        self.assertIsNotNone(resolved)
        command_display, handler_or_script, remaining = resolved
        self.assertEqual(command_display, "stimulus-predictions")
        self.assertEqual(handler_or_script, "scripts/export_stimulus_predictions.py")
        self.assertEqual(remaining, ["--output", "predictions.csv"])

    def test_nested_alpha_alias_resolution(self):
        resolved = cli.resolve_command(["alpha", "rt", "--joined-output", "joined.csv", "--summary-output", "summary.csv"])

        self.assertIsNotNone(resolved)
        command_display, handler_or_script, remaining = resolved
        self.assertEqual(command_display, "alpha rt")
        self.assertEqual(handler_or_script, "analyze_alpha_reaction_time.py")
        self.assertEqual(remaining, ["--joined-output", "joined.csv", "--summary-output", "summary.csv"])

    def test_nested_stimulus_alias_resolution(self):
        resolved = cli.resolve_command(["stimulus", "onset-scan", "--participants", "2"])

        self.assertIsNotNone(resolved)
        command_display, handler_or_script, remaining = resolved
        self.assertEqual(command_display, "stimulus onset-scan")
        self.assertEqual(handler_or_script, "scripts/export_stimulus_onset_scan.py")
        self.assertEqual(remaining, ["--participants", "2"])


if __name__ == "__main__":
    unittest.main()
