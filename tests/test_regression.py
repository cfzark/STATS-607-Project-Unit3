from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.benchmark import compare, run
from src.simulation import load_config


class RegressionTests(unittest.TestCase):
    def test_optimized_results_match_unit2_baseline(self):
        config = load_config("config.json")
        config.update(regimes=["linear", "fractional"], dimensions=[128],
                      theta=[0.6, 2.4], replicates=3)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            _, baseline = run(config, output, workers=1, reuse_parameters=False)
            _, serial = run(config, output, workers=1)
            _, parallel = run(config, output, workers=2)
        self.assertEqual(len(baseline), 12)
        compare(baseline, serial)
        compare(baseline, parallel)
