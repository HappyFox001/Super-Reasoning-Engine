import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def invoke(script, *args):
    return subprocess.run(
        [sys.executable, SCRIPTS / script, *args], capture_output=True, text=True,
    )


class GuardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        self.git("init", "-q")
        self.write("engine/kernel with spaces.py", "result = 1\n")
        self.write("sre/check/frozen check.py", "assert True\n")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-qm", "baseline")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", self.repo, *args], capture_output=True, text=True, check=True,
        ).stdout

    def write(self, path, content):
        file = self.repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")

    def guard(self, *args):
        result = invoke("guard.py", "--repo", self.repo, "--json", *args)
        self.assertTrue(result.stdout, result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_untracked_content_counts_and_reports_exact_filenames(self):
        tracked = "engine/kernel with spaces.py"
        untracked = "engine/candidate -> 新 kernel.py"
        self.write(tracked, "result = 1\n++ time.sleep(1)\n")
        self.write(untracked, "num_layers = 1\nresult = 2")
        self.write("sre/runs/run with spaces/events.jsonl", "pip install\n")
        index = self.git("ls-files", "--stage", "-z")

        code, report = self.guard("--allow", "engine/*", "--max-lines", "2")
        self.assertEqual(code, 1)
        self.assertEqual(report["verdict"], "VIOLATION")
        self.assertEqual(report["diff_lines"], 3)
        self.assertEqual(report["out_of_scope"], [])
        self.assertEqual({flag["file"] for flag in report["red_flags"]}, {tracked, untracked})
        self.assertEqual(self.git("ls-files", "--stage", "-z"), index)

        code, report = self.guard("--allow", "engine/*", "--max-lines", "3")
        self.assertEqual(code, 0)
        self.assertEqual(report["verdict"], "REVIEW")

    def test_checks_source_and_destination_of_rename(self):
        source = "sre/check/frozen check.py"
        destination = "engine/renamed check.py"
        self.git("mv", source, destination)
        code, report = self.guard("--protect", "sre/check/**", "--allow", "engine/*")
        self.assertEqual(code, 1)
        self.assertEqual(report["protected"], [source])
        self.assertEqual(report["out_of_scope"], [source])
        self.assertEqual(report["changed_files"], sorted([source, destination]))

    def test_staged_protected_change_is_detected_after_local_rollback(self):
        path = "sre/check/frozen check.py"
        self.write(path, "assert False\n")
        self.git("add", path)
        self.write(path, "assert True\n")
        code, report = self.guard("--protect", "sre/check/**")
        self.assertEqual(code, 1)
        self.assertEqual(report["protected"], [path])


class LedgerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.ledger = Path(directory.name) / "ledger.jsonl"

    def ledger_command(self, *args):
        return invoke("ledger.py", "--ledger", self.ledger, *args)

    def initialize(self, sol, noise, *args):
        result = self.ledger_command(
            "init", "--force", "--metric", "objective", "--sol-value", str(sol),
            "--noise", str(noise), *args,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.add(100, verdict="baseline")
        self.assertEqual(result.returncode, 0, result.stderr)

    def add(self, value, verdict="accept", *args):
        return self.ledger_command(
            "add", "--label", "candidate", "--verdict", verdict, "--value", str(value), *args,
        )

    def test_rejects_impossible_regression_and_noise_in_both_directions(self):
        cases = (
            (200, (), 210, 99, 103, 104),
            (50, ("--lower-is-better",), 45, 101, 97, 96),
        )
        for sol, options, impossible, regression, noise, valid in cases:
            with self.subTest(options=options):
                self.initialize(sol, 0.01, *options)
                before = self.ledger.read_bytes()
                for value, expected in ((impossible, "IMPOSSIBLE"), (regression, "REGRESSION"), (noise, "NOISE")):
                    with self.subTest(value=value):
                        result = self.add(value)
                        self.assertEqual(result.returncode, 2)
                        self.assertIn(expected, result.stderr)
                        self.assertEqual(self.ledger.read_bytes(), before)
                result = self.add(valid)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(self.ledger.read_text().splitlines()[-1])["value"], valid)

    def test_zero_noise_still_requires_improvement(self):
        self.initialize(200, 0)
        before = self.ledger.read_bytes()
        result = self.add(100)
        self.assertEqual(result.returncode, 2)
        self.assertIn("NOISE", result.stderr)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_accept_requires_a_baseline_and_noise_floor(self):
        result = self.ledger_command("init", "--metric", "objective", "--sol-value", "200")
        self.assertEqual(result.returncode, 0, result.stderr)
        for add_baseline in (False, True):
            with self.subTest(add_baseline=add_baseline):
                if add_baseline:
                    result = self.add(100, verdict="baseline")
                    self.assertEqual(result.returncode, 0, result.stderr)
                before = self.ledger.read_bytes()
                result = self.add(110, "accept", "--runs", "110,110")
                self.assertEqual(result.returncode, 2)
                self.assertIn("valid baseline and measured noise floor", result.stderr)
                self.assertEqual(self.ledger.read_bytes(), before)

        baseline = self.add(100, "baseline", "--runs", "99,100,101")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        accepted = self.add(110)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_model_mismatch_remains_advisory_and_rejects_remain_auditable(self):
        self.initialize(200, 0.01)
        outcome = self.add(104, "accept", "--predicted-gain", "0.5")
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertIn("MODEL-MISMATCH", outcome.stdout)
        rejected = self.add(210, "reject")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("IMPOSSIBLE", rejected.stdout)
        stored = json.loads(self.ledger.read_text().splitlines()[-1])
        self.assertEqual(stored["verdict"], "reject")

    def test_historical_invalid_accepts_are_visible_but_never_count(self):
        self.initialize(125, 0.01)
        with self.ledger.open("a", encoding="utf-8") as ledger:
            for n, value, flag in ((2, 125, "NOISE"), (3, 99, "REGRESSION"), (4, 130, "IMPOSSIBLE")):
                ledger.write(json.dumps({
                    "type": "iter", "n": n, "verdict": "accept", "value": value,
                    "label": "legacy candidate", "target": "kernel", "gain_vs_best": 0.005,
                    "sol_frac": value / 125, "flags": [flag],
                }) + "\n")
        result = self.ledger_command("show", "--stall")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("legacy candidate", result.stdout)
        self.assertIn("best accepted 100", result.stdout)
        self.assertNotIn("TARGET REACHED", result.stdout)
        self.assertNotIn("STALL:", result.stdout)
        self.assertNotIn("RABBIT-HOLE:", result.stdout)
        accepted = self.add(106)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_invalid_measurements_do_not_change_the_ledger(self):
        self.initialize(200, 0.01)
        before = self.ledger.read_bytes()
        for options in (("--value", "nan"), ("--value", "0"), ("--runs", "105,,106")):
            with self.subTest(options=options):
                result = self.ledger_command("add", "--label", "invalid", "--verdict", "accept", *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be a finite", result.stderr)
                self.assertEqual(self.ledger.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
