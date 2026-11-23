import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class QuickstartTests(unittest.TestCase):
    def calculate(self, directory, *args):
        output = directory / "sol.json"
        result = subprocess.run(
            [
                sys.executable, ROOT / "scripts" / "sol_calculator.py",
                "--model", ROOT / "examples" / "sol-quickstart" / "config.json",
                "--gpu", "H100-SXM5-80GB", "--weight-bits", "16",
                "--kv-bits", "16", "--compute-dtype", "bf16", "--tp", "1",
                "--prompt-len", "2048", "--gen-len", "512",
                "--batch-sweep", "1,32,64,128", "--tpot-slo-ms", "12",
                "--json", output, *args,
            ],
            cwd=directory, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(output.read_text(encoding="utf-8")), result.stdout

    def test_estimates_the_documented_theoretical_point(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.calculate(Path(directory))
        self.assertIsNone(result["achievable"])
        point = result["theoretical"]["operating_point"]
        self.assertEqual(point["B"], 86)
        self.assertEqual(point["bound"], "memory")
        self.assertLessEqual(point["tpot_ms"], 12)
        self.assertEqual(round(point["tok_s"]), 7168)

    def test_reports_each_ceiling_with_its_own_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            measured = directory / "hardware.json"
            measured.write_text('{"hbm_bw_Bps": 2.5e12}\n', encoding="utf-8")
            result, output = self.calculate(directory, "--measured", measured)
        theoretical = result["theoretical"]["operating_point"]
        achievable = result["achievable"]["operating_point"]
        self.assertLess(achievable["B"], theoretical["B"])
        for kind, point in (("theoretical", theoretical), ("achievable", achievable)):
            self.assertIn(
                f"Operating point B={point['B']}: {point['bound']}-bound, "
                f"TPOT {point['tpot_ms']:.2f} ms, {point['tok_s']:.0f} tok/s ({kind})",
                output,
            )


if __name__ == "__main__":
    unittest.main()
