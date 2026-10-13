"""Tests for tools/mtp_fetch.py's `verify` (#327): the pinned revision's tensors are checked against their sha256, a
corrupt install is reported (setup then fetches it again), and a verdict is kept by size and mtime.  The tensors are
made here, in a temporary folder - nothing is downloaded.

    python -m unittest tools.test_mtp_fetch
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import mtp_fetch as M  # noqa: E402

A, B = bytes(range(64)), bytes(range(100, 164)) * 2
HASHES = {"mtp.a": hashlib.sha256(A).hexdigest(), "mtp.b": hashlib.sha256(B).hexdigest()}


class Verify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.patch = mock.patch.object(M, "SHA256", HASHES)
        self.patch.start()
        (self.out / "tensors").mkdir()
        self.write("mtp.a", A)
        self.write("mtp.b", B)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def write(self, name, data):
        (self.out / "tensors" / (name + ".bin")).write_bytes(data)

    def verify(self):
        return M.verify(str(self.out))

    def test_honest_install_passes(self):
        self.assertEqual(self.verify(), [])

    def test_missing_and_wrong_tensors_are_named(self):
        (self.out / "tensors" / "mtp.b.bin").unlink()
        self.write("mtp.a", bytes(len(A)))              # right size, wrong bytes
        self.assertEqual(self.verify(), ["mtp.a", "mtp.b"])

    def test_verdict_is_kept_by_size_and_mtime(self):
        self.assertEqual(self.verify(), [])
        stamps = json.loads((self.out / "tensors" / "verified.json").read_text())
        self.assertEqual(sorted(stamps), ["mtp.a", "mtp.b"])
        with mock.patch.object(M, "sha256_of", side_effect=AssertionError("hashed an unchanged file")):
            self.assertEqual(self.verify(), [])

    def test_other_revision_has_nothing_to_check(self):
        with mock.patch.object(M, "REVISION", "main"):
            self.write("mtp.a", b"x")
            self.assertEqual(self.verify(), [])         # STRATA_MTP_REVISION: other hashes
        self.assertEqual(self.verify(), ["mtp.a"])

    def test_no_tensors_folder_has_nothing_to_check(self):
        self.assertEqual(M.verify(str(self.out / "nothing")), [])

    def test_cli_exit_code_3_for_bad_tensors(self):
        (self.out / "tensors" / "mtp.a.bin").unlink()
        with mock.patch.object(sys, "argv", ["mtp_fetch.py", "verify", "--out", str(self.out)]), \
                mock.patch("sys.stderr"), self.assertRaises(SystemExit) as cm:
            M.main()
        self.assertEqual(cm.exception.code, M.BAD)


class Pinned(unittest.TestCase):
    def test_every_tensor_has_a_hash(self):
        self.assertEqual(len(M.SHA256), 31)
        for name, h in M.SHA256.items():
            self.assertTrue(name.startswith("mtp."))
            self.assertRegex(h, "^[0-9a-f]{64}$")

    def test_modelscope_is_checked_against_the_pinned_hashes(self):
        with mock.patch.object(M, "MODELSCOPE", True), mock.patch.object(M, "REVISION", "main"):
            self.assertTrue(M.pinned())


class Setup(unittest.TestCase):
    def test_setup_rebuilds_a_corrupt_install(self):
        import setup
        with tempfile.TemporaryDirectory() as d:
            mtp = Path(d)
            self.assertFalse(setup.mtp_corrupt(mtp))              # no tensors kept: nothing to check
            (mtp / "tensors").mkdir()
            for code, want in ((0, False), (3, True), (1, False)):
                with self.subTest(code=code), mock.patch.object(
                        setup.subprocess, "run", return_value=mock.Mock(returncode=code)) as run:
                    self.assertEqual(setup.mtp_corrupt(mtp), want)
                    self.assertIn("verify", run.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
