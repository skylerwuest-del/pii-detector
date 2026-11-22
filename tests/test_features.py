"""Tests for readers, new PII types, score reuse, filters, reports, review, training extras and the CLI."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile
import torch
import config
import inference
import inspection
import main
import report
import review
import train
from generate_data import generate
from inference import Predictor, scan, query_results, folder_risk, read_meta, database_for, interrupted_scan
from inspection import inspect_file, count_indicators, decode_bytes, iban_valid, aba_valid
from model import PathCNN, model_settings
from preprocessing import PathEncoder, fold_accents, metadata_features, NUM_METADATA_FEATURES


def minimal_pdf(text):
    """A valid one-page PDF showing `text` (no external tools needed)."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "root"
        self.root.mkdir()
        self.output = self.base / "outputs"
        patches = [patch.object(config, "OUTPUT_DIRECTORY", self.output), patch.object(config, "DATA_DIRECTORY", self.base / "data"),
                   patch.object(config, "QUARANTINE_DIRECTORY", self.output / "quarantine"), patch.object(config, "SHOW_SCAN_PROGRESS", False),
                   patch.object(config, "TEST_DIRECTORY", self.base / "demo")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def checkpoint(self, metadata=False):
        settings = model_settings()
        settings["num_extra_features"] = NUM_METADATA_FEATURES if metadata else 0
        torch.manual_seed(3)
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "model.pt"
        torch.save(dict(model_settings=settings, state_dict=PathCNN(**settings).state_dict(),
                        encoder=PathEncoder().settings(), threshold=0.5), path)
        return path


class InspectionTests(Base):
    def test_ssn_structure_and_label(self):
        self.assertEqual(count_indicators("ref 000-12-3456")["SSN"], 0)       # Impossible area, unlabeled.
        self.assertEqual(count_indicators("SSN: 000-12-3456")["SSN"], 1)      # Labeled: counted.
        self.assertEqual(count_indicators("id 123-45-6789")["SSN"], 1)        # Plausible structure.
        self.assertEqual(count_indicators("x 666-12-3456 y 900-12-3456 z 123-00-4567")["SSN"], 0)

    def test_new_identifier_types(self):
        self.assertTrue(iban_valid("DE89 3704 0044 0532 0130 00"))
        self.assertFalse(iban_valid("DE00 3704 0044 0532 0130 00"))
        self.assertTrue(aba_valid("021000021"))
        self.assertFalse(aba_valid("123456789"))
        counts = count_indicators("Passport No: X1234567\npassport holders welcome\nDL: D1234-5678\n"
                                  "Routing: 021000021\nAccount #: 12345678\nDE89 3704 0044 0532 0130 00\n"
                                  "+44 20 7946 0958 and +1 202-555-0100")
        self.assertEqual(counts["Passport"], 1)
        self.assertEqual(counts["Driver's License"], 1)
        self.assertEqual(counts["Bank Account"], 2)
        self.assertEqual(counts["IBAN"], 1)
        self.assertEqual(counts["Phone"], 2)
        self.assertEqual(count_indicators("Routing: 123456789")["Bank Account"], 0)  # Fails the ABA checksum.

    def test_encodings(self):
        self.assertEqual(decode_bytes("Email".encode("utf-16"))[1], "utf-16")
        self.assertEqual(decode_bytes("Email: a@b.example".encode("utf-16-le"))[1], "utf-16-le")
        self.assertEqual(decode_bytes("Ren\xe9".encode("cp1252")), ("Ren\xe9", "cp1252"))
        self.assertEqual(decode_bytes("Ren\xe9".encode("utf-8"))[1], "utf-8")
        self.assertEqual(decode_bytes("Ren\xe9".encode("utf-8")[:-1], truncated=True)[1], "utf-8")
        path = self.root / "wide.txt"
        path.write_bytes("Email: a@example.invalid".encode("utf-16"))
        self.assertTrue(inspect_file(path, self.root)["indicators"]["Email"])

    def test_office_documents(self):
        with zipfile.ZipFile(self.root / "a.docx", "w") as z:
            z.writestr("word/document.xml", "<w:document><w:p><w:t>Name: Ann</w:t></w:p><w:p><w:t>a@example.invalid</w:t></w:p></w:document>")
        with zipfile.ZipFile(self.root / "b.xlsx", "w") as z:
            z.writestr("xl/sharedStrings.xml", "<sst><si><t>DOB: 1990-01-01</t></si></sst>")
        with zipfile.ZipFile(self.root / "c.pptx", "w") as z:
            z.writestr("ppt/slides/slide1.xml", "<p:sld><a:p><a:t>call 202-555-0100</a:t></a:p></p:sld>")
        self.assertEqual(inspect_file(self.root / "a.docx", self.root)["counts"], {"Email": 1, "Name": 1})
        self.assertEqual(inspect_file(self.root / "b.xlsx", self.root)["counts"], {"Date of Birth": 1})
        self.assertEqual(inspect_file(self.root / "c.pptx", self.root)["counts"], {"Phone": 1})
        (self.root / "broken.docx").write_text("not a zip", encoding="utf-8")
        self.assertEqual(inspect_file(self.root / "broken.docx", self.root)["status"], "Inspection failed")

    def test_zip_archive_members(self):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            z.writestr("deep.txt", "SSN 123-45-6789")
        with zipfile.ZipFile(self.root / "bundle.zip", "w") as z:
            z.writestr("notes/a.txt", "Email: a@example.invalid")
            z.writestr("image.bin", "123-45-6789")      # Not a readable member type.
            z.writestr("nested.zip", inner.getvalue())   # Nested archives are not opened.
        result = inspect_file(self.root / "bundle.zip", self.root)
        self.assertEqual(result["counts"], {"Email": 1})
        self.assertEqual(result["archive_members"], 1)

    def test_email_contacts_calendar(self):
        (self.root / "m.eml").write_bytes(b"From: Ann <ann@example.invalid>\nSubject: hi\nContent-Type: text/plain\n\ncall 202-555-0100\n")
        (self.root / "c.vcf").write_text("BEGIN:VCARD\nFN:Ann Example\nEND:VCARD\n", encoding="utf-8")
        (self.root / "e.ics").write_text("ATTENDEE;CN=Ann Example:mailto:x\n", encoding="utf-8")
        self.assertEqual(inspect_file(self.root / "m.eml", self.root)["counts"], {"Email": 1, "Phone": 1})
        self.assertEqual(inspect_file(self.root / "c.vcf", self.root)["counts"], {"Name": 1})
        self.assertEqual(inspect_file(self.root / "e.ics", self.root)["counts"], {"Name": 1})

    def test_pdf(self):
        path = self.root / "doc.pdf"
        path.write_bytes(minimal_pdf("Email: ann@example.invalid"))
        result = inspect_file(path, self.root)
        if inspection.pypdf is None:
            self.assertEqual(result["status"], "Inspection unsupported")
        else:
            self.assertEqual(result["counts"], {"Email": 1})


class ScanEngineTests(Base):
    def make_tree(self):
        (self.root / "hr").mkdir()
        (self.root / "hr" / "employees.csv").write_text("x", encoding="utf-8")
        (self.root / "hr" / "notes.txt").write_text("x", encoding="utf-8")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "manual.md").write_text("x", encoding="utf-8")

    def test_reuse_and_change_tracking(self):
        self.make_tree()
        predictor = Predictor(self.checkpoint())
        scan([self.root], predictor=predictor)
        first = read_meta(database_for(False))
        self.assertEqual(first["statuses"], {"new": 3, "changed": 0, "unchanged": 0})
        self.assertTrue(first["complete"])
        target = self.root / "hr" / "notes.txt"
        target.write_text("longer content", encoding="utf-8")
        os.utime(target, (time.time() + 5, time.time() + 5))
        with patch.object(predictor, "scores", wraps=predictor.scores) as scoring:
            scan([self.root], predictor=predictor)
            scored = [p for call in scoring.call_args_list for p in call.args[0]]
        self.assertEqual(scored, ["hr/notes.txt"])  # Only the changed file went through the CNN.
        meta = read_meta(database_for(False))
        self.assertEqual(meta["statuses"], {"new": 0, "changed": 1, "unchanged": 2})
        self.assertEqual(meta["reused"], 2)
        rows = query_results(database_for(False), changed=True)[0]
        self.assertEqual([Path(r[0]).name for r in rows], ["notes.txt"])

    def test_filters_and_folder_risk(self):
        self.make_tree()
        scan([self.root], predictor=Predictor(self.checkpoint()))
        database = database_for(False)
        self.assertEqual(query_results(database, extension="csv")[1], 1)
        self.assertEqual(query_results(database, extension=".MD")[1], 1)
        self.assertEqual(query_results(database, folder="hr")[1], 2)
        self.assertEqual(query_results(database, folder="h%")[1], 0)  # Wildcards are escaped.
        rows, total, _ = query_results(database, offset=1, limit=1)
        self.assertEqual((len(rows), total), (1, 3))
        folders = {r["folder"]: r for r in folder_risk(database)}
        self.assertEqual(folders["hr"]["files"], 2)
        self.assertEqual(folders["docs"]["files"], 1)

    def test_cancelled_scan_is_resumable(self):
        self.make_tree()
        predictor = Predictor(self.checkpoint())
        real = predictor.scores
        calls = []
        def flaky(paths, sizes=None):
            calls.append(paths)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real(paths, sizes)
        with patch.object(config, "INFERENCE_BATCH_SIZE", 1), patch.object(predictor, "scores", side_effect=flaky):
            scan([self.root], full=True, predictor=predictor)
        self.assertIsNotNone(interrupted_scan(full=True))
        self.assertEqual(read_meta(database_for(True))["files"], 1)
        scan([self.root], full=True, predictor=predictor)
        meta = read_meta(database_for(True))
        self.assertIsNone(interrupted_scan(full=True))
        self.assertEqual((meta["files"], meta["reused"]), (3, 1))

    def test_html_report_escapes_and_has_no_values(self):
        (self.root / "<i>x&.txt").write_text("Email: secret@example.invalid", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            inference.print_inspections(scan([self.root], predictor=Predictor(self.checkpoint())), approved=True)
        page = report.write_html_report().read_text(encoding="utf-8")
        self.assertIn("&lt;i&gt;x&amp;.txt", page)
        self.assertNotIn("<i>x", page)
        self.assertNotIn("secret@example.invalid", page)
        self.assertIn("Email x1", page)


class ReviewTests(Base):
    def write_report(self, names):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "inspection_report.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["absolute_path", "scan_root", "cnn_score", "status", "severity", "indicator_counts", "truncated"])
            for name in names:
                (self.root / name).write_text("Email: a@example.invalid", encoding="utf-8")
                writer.writerow([self.root / name, self.root, "0.9", "PII INDICATORS DETECTED", "Low", "Email=1", False])
            writer.writerow([self.root / "clean.txt", self.root, "0.1", "No supported PII indicators detected", "None", "", False])
        (self.root / "clean.txt").write_text("nothing", encoding="utf-8")

    def test_review_actions_and_feedback(self):
        self.write_report(["a.txt", "b.txt", "c.txt", "d.txt"])
        answers = iter(["k", "n", "q", "no", "q", "y", "s"])
        summary = review.review(ask=lambda _: next(answers))
        self.assertEqual(summary, {"kept": 1, "dismissed": 1, "quarantined": 1, "recycled": 0, "skipped": 1})
        self.assertFalse((self.root / "c.txt").exists())
        self.assertTrue((self.root / "d.txt").exists())
        labels = {e["path"]: e["label"] for e in review.load_feedback()}
        self.assertEqual(labels, {"a.txt": 1, "b.txt": 0, "c.txt": 1})
        entry = review.load_manifest()[0]
        review.restore(entry["id"])
        self.assertTrue((self.root / "c.txt").exists())
        with self.assertRaises(ValueError):
            review.restore(entry["id"])  # Already restored.

    def test_review_lists_only_flagged_unless_all(self):
        self.write_report(["a.txt"])
        self.assertEqual(len(review.load_review_items()), 1)
        self.assertEqual(len(review.load_review_items(include_clean=True)), 2)

    @unittest.skipIf(os.name == "nt" or os.uname().sysname == "Darwin", "freedesktop Trash layout")
    def test_trash_and_feedback_dedupe(self):
        file = self.root / "t.txt"
        file.write_text("x", encoding="utf-8")
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(self.base / "xdg")}):
            review.send_to_trash(file)
        self.assertTrue((self.base / "xdg" / "Trash" / "files" / "t.txt").exists())
        self.assertIn("Path=", (self.base / "xdg" / "Trash" / "info" / "t.txt.trashinfo").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            review.open_file(file)  # Gone: actions refuse missing files.
        review.record_feedback(self.root / "x.txt", self.root, 1)
        review.record_feedback(self.root / "x.txt", self.root, 0)
        self.assertEqual(review.load_feedback()[0]["label"], 0)
        self.assertEqual(len(review.load_feedback()), 1)


class TrainingExtrasTests(Base):
    def test_accents_and_metadata_features(self):
        self.assertEqual(fold_accents("nómina/employés"), "nomina/employes")
        folded = PathEncoder(strip_accents=True)
        self.assertEqual(folded.encode("Nómina.csv"), folded.encode("nomina.csv"))
        self.assertNotEqual(PathEncoder().encode("Nómina.csv"), PathEncoder().encode("nomina.csv"))
        features = metadata_features("a/b/report.xlsx", 50_000)
        self.assertEqual(len(features), NUM_METADATA_FEATURES)
        self.assertTrue(all(0.0 <= f <= 1.0 for f in features))
        self.assertEqual(metadata_features("x", 0)[2], 0.0)  # No extension.

    def test_calibration_helpers(self):
        logits = torch.tensor([4.0, 3.0, -3.0, -4.0])
        self.assertGreaterEqual(train.fit_temperature(logits, [1, 1, 0, 0]), 1.0)
        self.assertGreater(train.fit_temperature(torch.tensor([5.0, -5.0, 5.0, -5.0]), [1, 1, 0, 0]), 1.0)  # Wrong half: soften.
        self.assertEqual(train.calibration_error([1, 0], [1.0, 0.0])[0], 0.0)
        error, table = train.calibration_error([1, 0, 1, 0], [0.9, 0.9, 0.9, 0.9])
        self.assertAlmostEqual(error, 0.4)
        self.assertEqual(table[0]["count"], 4)

    def test_error_analysis(self):
        records = [dict(path="hr/payroll_1.csv", family=0), dict(path="docs/manual_1.txt", family=1), dict(path="hr/payroll_2.csv", family=0)]
        result = train.error_analysis(records, [1, 0, 1], [0, 1, 1], {0: "hr/payroll_{id}.csv", 1: "docs/manual_{id}.txt"})
        families = {f["family"]: f for f in result["families"]}
        self.assertEqual(families["hr/payroll_{id}.csv"]["missed"], 1)
        self.assertEqual(families["hr/payroll_{id}.csv"]["examples"], 2)
        self.assertEqual(families["docs/manual_{id}.txt"]["false_alarms"], 1)
        self.assertIn(("payroll", 1), result["missed_words"])

    def test_feedback_metadata_training_and_sweep(self):
        with patch.object(config, "EXAMPLES_PER_FAMILY", 2):
            generate()
        review.record_feedback(self.root / "my" / "clients.xlsx", self.root, 1, 4000)
        review.record_feedback(self.root / "games" / "clients.dat", self.root, 0, 10)
        with patch.object(config, "USE_METADATA_FEATURES", True), patch.object(config, "FEEDBACK_REPEAT", 2):
            results = train.run_training(4, 2, self.output / "model.pt", quiet=True)
        self.assertEqual(results["feedback_examples"], 2)
        self.assertTrue(results["metadata_features"])
        self.assertGreaterEqual(results["temperature"], 1.0)
        self.assertIn("errors", results)
        predictor = Predictor(self.output / "model.pt")
        self.assertTrue(predictor.uses_metadata)
        self.assertEqual(len(predictor.scores(["a.csv", "b.txt"], [10, 20])), 2)
        self.assertEqual(predictor.scores([]), [])
        runs = train.sweep(epochs=1, grid={"NUM_FILTERS": [8, 16]})
        self.assertEqual(len(runs), 2)
        self.assertEqual(config.NUM_FILTERS, 32)  # Restored after the sweep.
        self.assertTrue((self.output / "sweep.json").exists())


class InterfaceTests(Base):
    def test_parse_filters(self):
        self.assertEqual(main.parse_filters('--high --ext csv --folder "My Docs" --changed'),
                         dict(high=True, extension="csv", folder="My Docs", changed=True))
        self.assertEqual(main.parse_filters(""), dict(high=False, extension=None, folder=None, changed=False))
        with self.assertRaises(ValueError):
            main.parse_filters("--nope")

    def test_settings_validation_and_save(self):
        self.assertIsNone(main.parse_setting("MAX_FILES_TO_SCAN", "none"))
        self.assertEqual(main.parse_setting("MAX_FILES_TO_SCAN", "500"), 500)
        self.assertFalse(main.parse_setting("INCLUDE_HIDDEN_FILES", "off"))
        self.assertEqual(main.parse_setting("PREDICTION_THRESHOLD", "0.3"), 0.3)
        for name, bad in (("TOP_K_FILES", "-1"), ("PREDICTION_THRESHOLD", "2"), ("INCLUDE_HIDDEN_FILES", "maybe"),
                          ("MAX_INSPECTION_BYTES", "10"), ("TOP_K_FILES", "ten")):
            with self.assertRaises(ValueError):
                main.parse_setting(name, bad)
        settings_file = self.base / "user_settings.json"
        with patch.object(config, "USER_SETTINGS_FILE", settings_file), patch.object(config, "TOP_K_FILES", 3):
            # Only values that differ from config.py defaults are saved (this fixture also patches others).
            self.assertEqual(main.save_settings()["TOP_K_FILES"], 3)
            self.assertEqual(json.loads(settings_file.read_text(encoding="utf-8"))["TOP_K_FILES"], 3)
            answers = iter(["r", ""])
            main.settings_menu(ask=lambda _: next(answers))
            self.assertEqual(config.TOP_K_FILES, config.DEFAULTS["TOP_K_FILES"])
            self.assertFalse(settings_file.exists())

    def test_cli_scan_exit_codes_and_results(self):
        self.checkpoint()
        (self.root / "people.txt").write_text("Email: a@example.invalid", encoding="utf-8")
        self.assertEqual(main.cli(["scan", str(self.root), "--no-inspect", "--fail-on-pii"]), 0)  # Nothing opened.
        self.assertEqual(main.cli(["scan", str(self.root), "--fail-on-pii", "--top", "1"]), 3)
        self.assertEqual(main.cli(["results", "--ext", "txt"]), 0)
        with patch("main.webbrowser.open") as browser:
            self.assertEqual(main.cli(["report"]), 0)
            browser.assert_called_once()
        with self.assertRaises(SystemExit):
            main.cli(["scan", str(self.root / "missing")])

    def test_cli_requires_model_and_full_needs_approval(self):
        with self.assertRaises(SystemExit):
            main.cli(["scan", str(self.root)])
        self.checkpoint()
        with patch("main.fixed_drives", return_value=[self.root]), patch("main.scan") as scanning, \
                patch("builtins.input", return_value="no"):
            self.assertEqual(main.cli(["full"]), 1)
            scanning.assert_not_called()

    def test_results_pager(self):
        self.checkpoint()
        for i in range(5):
            (self.root / f"f{i}.txt").write_text("x", encoding="utf-8")
        scan([self.root], predictor=Predictor(self.output / "model.pt"))
        pager = main.ResultsPager()
        with patch.object(config, "RESULTS_PAGE_SIZE", 2), contextlib.redirect_stdout(io.StringIO()) as out:
            pager.show({}, full=False)
            pager.show(step=1)
            pager.show(step=1)
            pager.show(step=1)
        self.assertEqual(pager.offset, 4)
        self.assertIn("Already on the last page", out.getvalue())


if __name__ == "__main__":
    unittest.main()
