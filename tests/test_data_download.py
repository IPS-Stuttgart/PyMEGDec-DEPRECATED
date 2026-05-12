import os
import subprocess  # nosec B404
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pymegdec.data_download import download_meg_data_files

_TEST_ENV = {
    "BUSHMEG_WEBDAV_URL": "https://example.test/public.php/webdav/",
    "BUSHMEG_DATA_KEY": "key",
    "BUSHMEG_DATA_PASSWORD": "test-secret",  # nosec B105
}


def _completed(stdout=""):
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


class _FakeRclone:
    def __init__(self, listing):
        self.listing = listing
        self.calls = []

    def __call__(self, args, **_kwargs):
        self.calls.append(args)
        if args[1] == "obscure":
            return _completed("obscured-password\n")
        if args[1] == "lsf":
            return _completed(self.listing)
        if args[1] == "copyto":
            Path(args[3]).write_bytes(b"mat")
            return _completed()
        raise AssertionError(f"unexpected rclone command: {args}")


def _download_with_fake_rclone(fake_run, tmp_dir, selector_name, selector_value):
    manifest = Path(tmp_dir) / "manifest.txt"
    download_args = [
        "--source",
        "webdav-rclone",
        "--data-dir",
        str(Path(tmp_dir) / "data"),
        selector_name,
        selector_value,
        "--manifest",
        str(manifest),
    ]
    with patch.dict(os.environ, _TEST_ENV, clear=False), patch("pymegdec.data_download.subprocess.run", side_effect=fake_run):
        return download_meg_data_files(download_args), manifest


class DataDownloadTests(unittest.TestCase):
    def test_webdav_rclone_downloads_selected_file_indices_in_listing_order(self):
        fake_run = _FakeRclone("Part9Data.mat\nPart2CueData.mat\nPart2Data.mat\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code, manifest = _download_with_fake_rclone(fake_run, tmp_dir, "--file-indices", "2,3")

            self.assertEqual(exit_code, 0)
            copied_sources = [call[2] for call in fake_run.calls if call[1] == "copyto"]
            self.assertEqual(copied_sources, [":webdav:Part2CueData.mat", ":webdav:Part2Data.mat"])
            self.assertTrue((Path(tmp_dir) / "data" / "Part2CueData.mat").exists())
            self.assertTrue((Path(tmp_dir) / "data" / "Part2Data.mat").exists())
            self.assertIn("Part2CueData.mat", manifest.read_text(encoding="utf-8"))
            self.assertIn("Part2Data.mat", manifest.read_text(encoding="utf-8"))

    def test_webdav_rclone_rejects_out_of_range_file_indices(self):
        fake_run = _FakeRclone("only-file.mat\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(SystemExit):
                _download_with_fake_rclone(fake_run, tmp_dir, "--file-indices", "2")

    def test_webdav_rclone_downloads_selected_file_names_by_basename(self):
        fake_run = _FakeRclone("folder/Part2CueData.mat\nfolder/Part2Data.mat\nfolder/Part3Data.mat\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            exit_code, _manifest = _download_with_fake_rclone(fake_run, tmp_dir, "--file-names", "Part2CueData.mat,Part2Data.mat")

            self.assertEqual(exit_code, 0)
            copied_sources = [call[2] for call in fake_run.calls if call[1] == "copyto"]
            self.assertEqual(copied_sources, [":webdav:folder/Part2CueData.mat", ":webdav:folder/Part2Data.mat"])
            self.assertTrue((Path(tmp_dir) / "data" / "Part2CueData.mat").exists())
            self.assertTrue((Path(tmp_dir) / "data" / "Part2Data.mat").exists())

    def test_webdav_rclone_copy_timeout_is_reported(self):
        def fake_run(args, **kwargs):
            if args[1] == "obscure":
                return _completed("obscured-password\n")
            if args[1] == "lsf":
                return _completed("Part2Data.mat\n")
            if args[1] == "copyto":
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            raise AssertionError(f"unexpected rclone command: {args}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(SystemExit, "rclone copy 'Part2Data.mat' timed out after 1800 seconds"):
                _download_with_fake_rclone(fake_run, tmp_dir, "--file-names", "Part2Data.mat")


if __name__ == "__main__":
    unittest.main()
