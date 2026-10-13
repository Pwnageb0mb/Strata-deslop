"""How many jobs a build runs: setup.build_jobs() - half the cores by default, STRATA_BUILD_JOBS overrides it.

    python -m unittest tools.test_setup_build_jobs
"""
from __future__ import annotations

import os
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class BuildJobs(unittest.TestCase):
    def jobs(self, value, cores=28):
        with unittest.mock.patch.dict(os.environ, {} if value is None else {"STRATA_BUILD_JOBS": value}), \
             unittest.mock.patch.object(setup.os, "cpu_count", return_value=cores):
            return setup.build_jobs()

    def test_default_is_half_the_cores(self):
        self.assertEqual(self.jobs(None, 28), 14)
        self.assertEqual(self.jobs("", 28), 14)

    def test_all_uses_every_core(self):
        self.assertEqual(self.jobs("all", 28), 28)
        self.assertEqual(self.jobs("ALL", 28), 28)
        self.assertEqual(self.jobs("0", 28), 28)

    def test_a_number_is_used_as_given(self):
        self.assertEqual(self.jobs("28", 28), 28)
        self.assertEqual(self.jobs("3", 28), 3)

    def test_nonsense_falls_back_to_the_default(self):
        self.assertEqual(self.jobs("banana", 28), 14)

    def test_never_zero_or_negative(self):
        self.assertEqual(self.jobs("0", 1), 2)
        self.assertEqual(self.jobs("-4", 28), 1)


if __name__ == "__main__":
    unittest.main()
