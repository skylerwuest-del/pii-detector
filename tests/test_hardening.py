"""Race-safe opening, legacy Office/Outlook files, OCR, schedules and Windows-only behavior."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import inspection
import review
import scanner
from inspection import inspect_file, count_indicators, OLE_MAGIC


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "root"
        self.root.mkdir()
        self.file = self.root / "a.txt"
        self.file.write_text("Email: a@example.invalid", encoding="utf-8")
        (self.root / "other.txt").write_text("different file", encoding="utf-8")

    # ----- race-safe opening -----
    def test_safe_open_reads_regular_file(self):
        with scanner.safe_open(self.file, self.root) as handle:
            self.assertEqual(handle.read(), b"Email: a@example.invalid")

    def test_link_swapped_in_after_validation_is_refused(self):
        link = self.root / "link.txt"
        try:
            link.symlink_to(self.root / "other.txt")
        except OSError:
            self.skipTest("Symlinks unavailable")
        # Simulate the race: validation saw a safe path, but a link is there when we open.
        with patch("scanner.safe_path", return_value=link):
            with self.assertRaises(ValueError):
                scanner.safe_open(link, self.root)

    def test_file_replaced_between_check_and_open_is_refused(self):
        real = os.lstat
        other = self.root / "other.txt"
        swapped = lambda p, *a, **k: real(other) if str(p) == str(self.file) else real(p, *a, **k)
        with patch("scanner.os.lstat", side_effect=swapped):
            with self.assertRaises(ValueError):
                scanner.safe_open(self.file, self.root)

    def test_inspection_reports_failure_for_swapped_file(self):
        real = os.lstat
        other = self.root / "other.txt"
        swapped = lambda p, *a, **k: real(other) if str(p) == str(self.file) else real(p, *a, **k)
        with patch("scanner.os.lstat", side_effect=swapped):
            self.assertEqual(inspect_file(self.file, self.root)["status"], "Inspection failed")

    # ----- legacy Office / Outlook -----
    def test_legacy_ole_strings(self):
        body = (OLE_MAGIC + b"\x00" * 64 + "Email: a@example.invalid".encode("utf-16-le") + b"\x00\x00"
                + b"\rName: Ann Example\rSSN: 123-45-6789\r" + bytes(range(1, 30)))
        for extension in (".doc", ".xls", ".ppt", ".msg"):
            path = self.root / f"old{extension}"
            path.write_bytes(body)
            result = inspect_file(path, self.root)
            self.assertEqual(result["counts"], {"Email": 1, "SSN": 1, "Name": 1}, extension)
            self.assertEqual(result["encoding"], "ole-strings")
        (self.root / "fake.doc").write_bytes(b"plain text pretending to be a doc")
        self.assertEqual(inspect_file(self.root / "fake.doc", self.root)["status"], "Inspection failed")

    # ----- schedules -----
    def test_schedule_lines_and_calendar_events(self):
        text = ("2025-03-04 3:30 pm dentist appointment for Sam\n"
                "Team standup 2025-03-04 9:00\n"           # Not personal: no appointment word.
                "Mar 5 10:00 am pick up kids from school\n"
                "dentist appointment soon\n")               # No date or time.
        self.assertEqual(count_indicators(text)["Schedule"], 2)
        calendar = ("BEGIN:VEVENT\nDTSTART:20250304T153000\nLOCATION:Clinic\nEND:VEVENT\n"
                    "BEGIN:VEVENT\nDTSTART:20250305T100000\nSUMMARY:Holiday\nEND:VEVENT\n")
        self.assertEqual(count_indicators(calendar)["Schedule"], 1)
        path = self.root / "plan.ics"
        path.write_text(calendar, encoding="utf-8")
        self.assertEqual(inspect_file(path, self.root)["severity"], "Low")

    # ----- OCR (runs only where pytesseract, Pillow and Tesseract are installed) -----
    def render(self, path, lines):
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGB", (900, 80 + 60 * len(lines)), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=36)
        for index, line in enumerate(lines):
            draw.text((20, 20 + 60 * index), line, fill="black", font=font)
        image.save(path)

    @unittest.skipUnless(inspection.OCR, "OCR not installed")
    def test_ocr_image(self):
        self.render(self.root / "photo.png", ["Phone: 202-555-0100", "Email: ann@example.com"])
        result = inspect_file(self.root / "photo.png", self.root)
        self.assertEqual(result["encoding"], "ocr")
        self.assertEqual(result["counts"], {"Email": 1, "Phone": 1})

    @unittest.skipUnless(inspection.OCR and inspection.pypdf, "OCR or pypdf not installed")
    def test_scanned_pdf_uses_ocr(self):
        self.render(self.root / "scan.pdf", ["Phone: 202-555-0100"])  # Pillow writes an image-only PDF.
        result = inspect_file(self.root / "scan.pdf", self.root)
        self.assertEqual(result["encoding"], "pdf+ocr")
        self.assertEqual(result["counts"], {"Phone": 1})

    def test_images_unsupported_without_ocr(self):
        (self.root / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        with patch.object(inspection, "READABLE_EXTENSIONS", inspection.READABLE_EXTENSIONS - inspection.IMAGE_EXTENSIONS):
            self.assertEqual(inspect_file(self.root / "photo.png", self.root)["status"], "Inspection unsupported")

    # ----- Windows-only (exercised by CI on windows-latest) -----
    @unittest.skipUnless(os.name == "nt", "Windows Recycle Bin")
    def test_windows_recycle_bin(self):
        target = self.root / "recycle me.txt"
        target.write_text("synthetic", encoding="utf-8")
        self.assertEqual(review.send_to_trash(target), "Recycle Bin")
        self.assertFalse(target.exists())

    @unittest.skipUnless(os.name == "nt", "Windows handle path check")
    def test_windows_final_path_check(self):
        with scanner.safe_open(self.file, self.root) as handle:
            self.assertTrue(scanner.within(scanner._final_path_of_handle(handle.fileno()), self.root))
        outside = Path(self.temp.name).resolve() / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("synthetic", encoding="utf-8")
        junction = self.root / "junction"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], check=True, capture_output=True)
        try:
            with self.assertRaises(ValueError):
                scanner.safe_open(junction / "secret.txt", self.root)
        finally:
            junction.rmdir()


if __name__ == "__main__":
    unittest.main()
