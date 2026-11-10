"""Standard-library tests: python -m unittest discover -s tests -v."""
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import torch
import config
from preprocessing import PathEncoder
from model import PathCNN, model_settings
from scanner import Navigator, enumerate_files, safe_path, fixed_drives
from inspection import inspect_file, inspect_candidates
from inference import Predictor, scan
from generate_data import generate
from main import full_mode


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "authorized"
        self.root.mkdir()
        self.file = self.root / "Employee records !.txt"
        self.file.write_text("Name: Example Person\nEmail: fake@example.invalid\nPhone: 202-555-0100\nSSN: 000-12-3456", encoding="utf-8")

    def make_predictor(self):
        checkpoint = self.base / "model.pt"
        settings = model_settings()
        torch.manual_seed(7)
        torch.save(dict(model_settings=settings, state_dict=PathCNN(**settings).state_dict(),
                        encoder=PathEncoder().settings()), checkpoint)
        return Predictor(checkpoint)

    def test_preprocessing(self):
        encoder = PathEncoder(max_length=12)
        self.assertEqual(encoder.encode("ABC\\x.txt"), encoder.encode("abc/x.txt"))
        self.assertEqual(len(encoder.encode("x")), 12)
        self.assertIn(1, encoder.encode("未知"))
        self.assertEqual(encoder.encode("a/" * 100 + "file.txt"), encoder.encode(("a/" * 100 + "file.txt")[-12:]))
        self.assertTrue(all(0 <= i < config.VOCABULARY_SIZE for i in encoder.encode("x")))

    def test_cnn_shape_and_gradient(self):
        model = PathCNN(**model_settings())
        ids = torch.tensor([PathEncoder().encode("folder/file.txt")] * 3)
        logits = model(ids)
        self.assertEqual(tuple(logits.shape), (3,))
        initial = model.conv.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        torch.nn.BCEWithLogitsLoss()(logits, torch.ones(3)).backward()
        optimizer.step()
        self.assertFalse(torch.equal(initial, model.conv.weight))

    def test_navigation_lock(self):
        child = self.root / "sub folder"
        child.mkdir()
        nav = Navigator(self.root)
        nav.select()
        with self.assertRaises(ValueError):
            nav.cd("..")
        with self.assertRaises(ValueError):
            nav.cd(str(self.base))
        nav.cd("sub folder")
        nav.cd("..")
        self.assertEqual(nav.current, self.root)
        with self.assertRaises(ValueError):
            nav.select()
        with self.assertRaises(ValueError):
            safe_path(self.base / "outside.txt", nav.root)

    def test_escaping_link(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("synthetic", encoding="utf-8")
        link = self.root / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                self.skipTest("Symlinks unavailable")
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], check=True, capture_output=True)
        nav = Navigator(self.root)
        nav.select()
        with self.assertRaises(ValueError):
            nav.cd("escape")
        self.assertEqual([p for p, _ in enumerate_files(self.root)], [self.file])
        # Remove only the link itself; never recurse through the target.
        if link.is_symlink():
            link.unlink()
        else:
            link.rmdir()

    def test_unreadable_directory(self):
        with patch("scanner.os.scandir", side_effect=PermissionError):
            self.assertEqual(list(enumerate_files(self.root)), [])
        with patch("scanner.is_link", side_effect=PermissionError):
            self.assertEqual(list(enumerate_files(self.root)), [])

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_windows_junction_boundary(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "not_scanned.txt").write_text("synthetic", encoding="utf-8")
        junction = self.root / "junction"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], check=True, capture_output=True)
        try:
            nav = Navigator(self.root)
            nav.select()
            with self.assertRaises(ValueError):
                nav.cd("junction")
            with self.assertRaises(ValueError):
                safe_path(junction / "not_scanned.txt", self.root)
            self.assertEqual([p for p, _ in enumerate_files(self.root)], [self.file])
        finally:
            junction.rmdir()  # Remove the junction itself, not its target.

    @unittest.skipUnless(os.name == "nt", "Windows drive discovery test")
    def test_fixed_drive_discovery(self):
        with patch("scanner.ctypes.windll.kernel32.GetLogicalDrives", return_value=13), patch("scanner.ctypes.windll.kernel32.GetDriveTypeW", side_effect=[3, 4, 2]):
            self.assertEqual(fixed_drives(), [Path("A:/")])

    def test_inspection_categories_and_no_values(self):
        result = inspect_file(self.file, self.root)
        self.assertTrue(all(result["indicators"].values()))
        self.assertNotIn("fake@example.invalid", str(result))
        self.file.write_text("General documentation only", encoding="utf-8")
        self.assertEqual(inspect_file(self.file, self.root)["status"], "No supported PII indicators detected")
        self.file.unlink()
        self.assertEqual(inspect_file(self.file, self.root)["status"], "Inspection failed")

    def test_csv_and_gitignore(self):
        path = self.root / "people.csv"
        path.write_text("full_name,email\nExample Person,x@example.invalid\n", encoding="utf-8")
        result = inspect_file(path, self.root)
        self.assertTrue(result["indicators"]["Name"])
        self.assertTrue(result["indicators"]["Email"])
        ignore = self.root / ".gitignore"
        ignore.write_text("private/payroll.csv\n", encoding="utf-8")
        self.assertEqual(inspect_file(ignore, self.root)["status"], "Metadata clues detected")

    def test_unsupported_not_opened_and_no_approval(self):
        unsupported = self.root / "file.bin"
        with patch.object(Path, "open", side_effect=AssertionError("Must not read")):
            self.assertEqual(inspect_file(unsupported, self.root)["status"], "Inspection unsupported")
            self.assertEqual(inspect_candidates([dict(path=self.file, root=self.root)], False), [])

    def test_both_modes_same_checkpoint_and_absolute_reports(self):
        predictor = self.make_predictor()
        (self.root / "unsupported.bin").write_bytes(b"synthetic")
        output = self.base / "outputs"
        with patch.object(config, "OUTPUT_DIRECTORY", output), patch.object(config, "SHOW_SCAN_PROGRESS", False):
            with patch("inspection.inspect_file", side_effect=AssertionError("Metadata scan must not read contents")):
                controlled = scan([self.root], predictor=predictor)
                full = scan([self.root], full=True, predictor=predictor)
            self.assertEqual(list(controlled), list(full))
            with (output / "full_scan_results.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(Path(r["absolute_path"]).is_absolute() for r in rows))
            self.assertIn(str(self.file), (output / "file_paths.txt").read_text(encoding="utf-8"))
            for row in rows:
                relative = Path(row["absolute_path"]).relative_to(self.root).as_posix()
                self.assertEqual(len(predictor.encoder.encode(relative)), config.MAX_PATH_LENGTH)
                self.assertAlmostEqual(float(row["cnn_score"]), predictor.scores([relative])[0], places=6)

    def test_full_mode_approval_gate(self):
        candidate = dict(path=str(self.file), root=str(self.root), score=0.9)
        with patch("main.fixed_drives", return_value=[self.root]), patch("main.scan", return_value=[candidate]) as scanning, patch("builtins.input", side_effect=["SCAN", "no"]), patch("inspection.inspect_file") as inspection:
            full_mode()
            scanning.assert_called_once()
            inspection.assert_not_called()
        with patch("main.fixed_drives", return_value=[self.root]), patch("main.scan") as scanning, patch("builtins.input", return_value="no"):
            full_mode()
            scanning.assert_not_called()

    def test_limits_exclusions_and_failed_inspection(self):
        excluded = self.root / "skip"
        excluded.mkdir()
        (excluded / "hidden.txt").write_text("synthetic", encoding="utf-8")
        self.assertEqual([p for p, _ in enumerate_files(self.root, [excluded])], [self.file])
        with patch.object(config, "OUTPUT_DIRECTORY", self.base / "output"), patch.object(config, "MAX_FILES_TO_SCAN", 1):
            scan([self.root], full=True, predictor=self.make_predictor())
            rows = (self.base / "output" / "full_scan_file_paths.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
        candidates = [dict(path="missing.txt", root=self.root), dict(path=self.file, root=self.root)]
        self.assertEqual(len(inspect_candidates(candidates, True)), 2)

    def test_group_split_and_reproducibility(self):
        with patch.object(config, "DATA_DIRECTORY", self.base / "data"), patch.object(config, "TEST_DIRECTORY", self.base / "demo"):
            first = generate()
            self.assertEqual(first, generate())
            groups = [set(r["family"] for r in records) for records in first.values()]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            for split, records in first.items():
                for record in records:
                    content = (config.DATA_DIRECTORY / "corpus" / split / record["path"]).read_text(encoding="utf-8")
                    self.assertEqual("example@example.invalid" in content, bool(record["label"]))


if __name__ == "__main__":
    unittest.main()
