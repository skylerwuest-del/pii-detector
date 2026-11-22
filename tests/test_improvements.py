"""Tests for training methods, scanner statistics, richer inspection and the terminal helpers."""
import contextlib
import csv
import io
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import torch
import config
import ui
from generate_data import generate
from inference import Predictor, print_inspections
from inspection import inspect_file, luhn_valid, severity
from model import PathCNN, model_settings
from preprocessing import PathEncoder
from scanner import Navigator, enumerate_files
from train import augment_path, metrics, roc_auc, run_training, tune_threshold


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "root"
        self.root.mkdir()

    # ----- inspection -----
    def test_luhn_and_card_detection(self):
        self.assertTrue(luhn_valid("4111 1111 1111 1111"))
        self.assertFalse(luhn_valid("1234 5678 9012 3456"))
        self.assertFalse(luhn_valid("123"))
        path = self.root / "cards.txt"
        path.write_text("pay 4111 1111 1111 1111 now\nnot a card 1234 5678 9012 3456\n", encoding="utf-8")
        result = inspect_file(path, self.root)
        self.assertEqual(result["counts"], {"Credit Card": 1})
        self.assertNotIn("4111", str(result))  # Counts only, never matched values.

    def test_labeled_dob_and_address(self):
        path = self.root / "profile.md"
        path.write_text("DOB: 1990-02-03\nLives at 12 Oak Street\n", encoding="utf-8")
        result = inspect_file(path, self.root)
        self.assertTrue(result["indicators"]["Date of Birth"])
        self.assertTrue(result["indicators"]["Address"])

    def test_tsv_columns_not_double_counted(self):
        path = self.root / "people.tsv"
        path.write_text("ssn\tnote\n123-45-6789\thello\n", encoding="utf-8")
        self.assertEqual(inspect_file(path, self.root)["counts"], {"SSN": 1})
        empty = self.root / "empty.csv"
        empty.write_text("email,phone\n,\n", encoding="utf-8")
        self.assertEqual(inspect_file(empty, self.root)["counts"], {})

    def test_severity_bands(self):
        self.assertEqual(severity([]), "None")
        self.assertEqual(severity(["Email"]), "Low")
        self.assertEqual(severity(["Email", "Phone"]), "Medium")
        self.assertEqual(severity(["SSN"]), "High")
        self.assertEqual(severity(["Email", "Phone", "Name"]), "High")

    def test_new_text_extension_supported_and_unlisted_not(self):
        log = self.root / "server.log"
        log.write_text("contact: x@example.invalid", encoding="utf-8")
        self.assertTrue(inspect_file(log, self.root)["indicators"]["Email"])
        with patch.object(config, "SUPPORTED_EXTENSIONS", [".txt"]):
            self.assertEqual(inspect_file(log, self.root)["status"], "Inspection unsupported")

    # ----- scanner -----
    def test_enumeration_stats_and_exclusions(self):
        (self.root / "keep.txt").write_text("x", encoding="utf-8")
        (self.root / ".hidden.txt").write_text("x", encoding="utf-8")
        skipped = self.root / "skipme"
        skipped.mkdir()
        (skipped / "a.txt").write_text("x", encoding="utf-8")
        stats = {}
        with patch.object(config, "INCLUDE_HIDDEN_FILES", False):
            found = [p.name for p, _ in enumerate_files(self.root, [skipped], stats)]
        self.assertEqual(found, ["keep.txt"])
        self.assertEqual(stats, {"hidden": 1, "excluded": 1})
        # A sibling that merely shares a prefix with an exclusion must not be excluded.
        sibling = self.root / "skipme_not"
        sibling.mkdir()
        (sibling / "b.txt").write_text("x", encoding="utf-8")
        names = [p.name for p, _ in enumerate_files(self.root, [skipped])]
        self.assertIn("b.txt", names)

    def test_navigator_listing_tree_and_previous(self):
        (self.root / "folder").mkdir()
        (self.root / "folder" / "inner.txt").write_text("abc", encoding="utf-8")
        (self.root / "a.txt").write_text("abc", encoding="utf-8")
        nav = Navigator(self.root)
        self.assertEqual([(n, d) for n, d, _, _ in nav.listing()], [("folder", True), ("a.txt", False)])
        self.assertEqual(nav.tree_lines(2), ["|-- folder/", "|   `-- inner.txt", "`-- a.txt"])
        self.assertEqual(len(nav.tree_lines(1)), 2)
        with self.assertRaises(ValueError):
            nav.cd("-")  # No previous directory yet.
        nav.cd("folder")
        nav.cd("-")
        self.assertEqual(nav.current, self.root)
        self.assertLessEqual(len(nav.tree_lines(5, limit=1)), 2)

    # ----- training methods -----
    def test_augmentation_is_deterministic_and_optional(self):
        path = "hr/employee_list.csv"
        first = augment_path(path, random.Random(3))
        self.assertEqual(first, augment_path(path, random.Random(3)))
        with patch.object(config, "AUGMENT_PROBABILITY", 0.0):
            self.assertEqual(augment_path(path, random.Random(3)), path)
        with patch.object(config, "AUGMENT_PROBABILITY", 1.0):
            changed = {augment_path(path, random.Random(s)) for s in range(20)}
            self.assertGreater(len(changed), 5)  # Varied, not a fixed rewrite.

    def test_metrics_auc_and_threshold(self):
        labels = [1, 1, 0, 0]
        result = metrics(labels, [1, 0, 1, 0])
        self.assertEqual(result["confusion"], dict(tp=1, fp=1, fn=1, tn=1))
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertEqual(roc_auc(labels, [0.9, 0.8, 0.2, 0.1]), 1.0)
        self.assertEqual(roc_auc(labels, [0.1, 0.2, 0.8, 0.9]), 0.0)
        self.assertEqual(roc_auc(labels, [0.5] * 4), 0.5)
        self.assertTrue(math.isnan(roc_auc([1, 1], [0.1, 0.2])))
        threshold, score = tune_threshold(labels, [0.9, 0.8, 0.2, 0.1], 1.0)
        self.assertTrue(0.2 < threshold <= 0.8)
        self.assertEqual(score, 1.0)
        # Favoring recall (beta 2) never picks a higher cutoff than favoring precision.
        scores, truth = [0.9, 0.6, 0.55, 0.4, 0.3], [1, 1, 0, 1, 0]
        self.assertLessEqual(tune_threshold(truth, scores, 2.0)[0], tune_threshold(truth, scores, 0.5)[0])

    def checkpoint_with(self, threshold):
        settings = model_settings()
        saved = dict(model_settings=settings, state_dict=PathCNN(**settings).state_dict(), encoder=PathEncoder().settings())
        if threshold is not None:
            saved["threshold"] = threshold
        path = self.base / "m.pt"
        torch.save(saved, path)
        return path

    def test_predictor_threshold_source(self):
        self.assertAlmostEqual(Predictor(self.checkpoint_with(0.31)).threshold, 0.31)
        self.assertEqual(Predictor(self.checkpoint_with(None)).threshold, config.PREDICTION_THRESHOLD)
        with patch.object(config, "AUTO_TUNE_THRESHOLD", False):
            self.assertEqual(Predictor(self.checkpoint_with(0.31)).threshold, config.PREDICTION_THRESHOLD)

    def test_short_training_run_and_shuffled_control(self):
        with patch.object(config, "DATA_DIRECTORY", self.base / "data"), patch.object(config, "TEST_DIRECTORY", self.base / "demo"), \
                patch.object(config, "EXAMPLES_PER_FAMILY", 2), contextlib.redirect_stdout(io.StringIO()):
            generate()
            checkpoint = self.base / "out" / "model.pt"
            results = run_training(5, 3, checkpoint, quiet=True)
            self.assertTrue(checkpoint.is_file())
            self.assertGreater(results["max_conv_weight_change"], 0)
            self.assertEqual(results["epochs_run"], len(results["history"]))
            self.assertTrue(0.0 <= results["threshold"] <= 1.0)
            self.assertEqual(sum(results["confusion"].values()), results["test_examples"])
            self.assertAlmostEqual(Predictor(checkpoint).threshold, results["threshold"], places=6)
            control = run_training(5, 2, shuffle_labels=True, quiet=True)
            self.assertTrue(control["shuffled_labels"])
            # Same seed and data reproduce the same run.
            self.assertEqual(run_training(5, 3, quiet=True)["history"], results["history"])

    def test_early_stopping(self):
        with patch.object(config, "DATA_DIRECTORY", self.base / "data"), patch.object(config, "TEST_DIRECTORY", self.base / "demo"), \
                patch.object(config, "EXAMPLES_PER_FAMILY", 2), patch.object(config, "EARLY_STOPPING_PATIENCE", 1), \
                patch.object(config, "LEARNING_RATE", 0.0), contextlib.redirect_stdout(io.StringIO()):
            generate()
            results = run_training(1, 10, quiet=True)
        # With a frozen model the first epoch is "best" and the second triggers the stop.
        self.assertTrue(results["stopped_early"])
        self.assertLess(results["epochs_run"], 10)

    # ----- reports and terminal helpers -----
    def test_inspection_report_written_and_declined(self):
        file = self.root / "people.txt"
        file.write_text("Name: Example Person\nEmail: a@example.invalid\n", encoding="utf-8")
        candidates = [dict(path=str(file), root=str(self.root), score=0.9)]
        output = self.base / "reports"
        with patch.object(config, "OUTPUT_DIRECTORY", output), contextlib.redirect_stdout(io.StringIO()) as out:
            print_inspections(candidates, approved=False)
            self.assertFalse((output / "inspection_report.csv").exists())
            print_inspections(candidates, approved=True)
        rows = list(csv.DictReader((output / "inspection_report.csv").open(encoding="utf-8")))
        self.assertEqual(len(rows), 1)
        self.assertIn("Email=1", rows[0]["indicator_counts"])
        self.assertNotIn("a@example.invalid", out.getvalue() + (output / "inspection_report.csv").read_text(encoding="utf-8"))

    def test_ui_helpers(self):
        self.assertEqual(ui.shorten("abcdefghij", 6), "...hij")
        self.assertEqual(ui.human_size(1536), "1.5 KB")
        self.assertEqual(ui.human_time(75), "1m15s")
        self.assertEqual(ui.bar(0.5, 4), "[##..]")
        self.assertEqual(ui.bar(7, 4), "[####]")  # Clamped.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ui.table(["A", "B"], [["x", 1], ["yy", 22]], aligns=["<", ">"])
            ui.line_chart({"t": [3, 2, 1], "v": [1, 2, 3]}, height=3, columns=10)
            progress = ui.Progress("quiet", enabled=False)
            progress.update(5)
            progress.finish()
        self.assertIn("yy", out.getvalue())
        self.assertNotIn("quiet", out.getvalue())
        self.assertNotIn("\033", out.getvalue())  # Not a terminal: no color codes.


if __name__ == "__main__":
    unittest.main()
