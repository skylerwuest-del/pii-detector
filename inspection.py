"""Deterministic verification rules, not another ML algorithm. Never return matched values.

Readers turn a supported file into bounded text (plain text in several encodings,
Office documents, PDFs when pypdf is installed, zip archives, email, calendar and
contact files); counting rules then report how many indicators of each category
appear. Matched values are counted and discarded.
"""
import csv
import email
from email import policy
import io
import os
import re
import shutil
import zipfile
from pathlib import Path
import config
from scanner import safe_open

try:  # Optional: PDF text extraction. Without it PDFs are ranked but not opened.
    import pypdf
except ImportError:  # pragma: no cover - depends on the environment
    pypdf = None


def _ocr_engine():
    """pytesseract + Pillow + the Tesseract program, or None when any piece is missing."""
    if not getattr(config, "OCR_ENABLED", False):
        return None
    try:
        import pytesseract
        from PIL import Image
        # The Windows installer does not add Tesseract to PATH; use its standard location.
        default_windows = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe"
        if config.TESSERACT_COMMAND:
            pytesseract.pytesseract.tesseract_cmd = str(config.TESSERACT_COMMAND)
        elif os.name == "nt" and not shutil.which("tesseract") and default_windows.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(default_windows)
        pytesseract.get_tesseract_version()  # Raises when the Tesseract program is not installed.
        return pytesseract, Image
    except Exception:
        return None


OCR = _ocr_engine()

NAME_PATTERN = (r"(?im)^\s*(?:name|full[ _]name|employee[ _]name|first[ _]name|last[ _]name)\s*:\s*\S+"
                r"|^(?:FN|N)(?:;[^:\n]*)?:\s*[^\s;]"  # vCard name lines.
                r"|\bCN=[^;:\n\"]*[A-Za-z][^;:\n\"]*")  # Calendar attendee/organizer names.
PATTERNS = {
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "Name": NAME_PATTERN,
    "Date of Birth": r"(?i)\b(?:dob|date[ _]of[ _]birth|birth[ _]?date|birthday)\b\s*[:=]\s*\S+",
    "Address": r"(?i)\b\d{1,5}\s+(?:[A-Za-z0-9.'-]+\s+){1,4}(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b\.?",
    # Labeled identifiers: the value must contain a digit so "passport holders" is not a hit.
    "Passport": r"(?i)\bpassport(?:\s*(?:no|number|num|#))?\.?\s*[:=#]?\s*(?=[A-Z0-9]*\d)[A-Z0-9]{6,9}\b",
    "Driver's License": r"(?i)\b(?:driver'?s?\s+licen[cs]e|DL)(?:\s*(?:no|number|#))?\.?\s*[:=#]\s*(?=[A-Z0-9-]*\d)[A-Z0-9-]{5,15}\b",
}
US_PHONE = re.compile(r"(?<!\d)(?:\+1[ -]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
INTL_PHONE = re.compile(r"(?<![\w+])\+(?!1[ .(-])[2-9]\d{0,2}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?!\d)")
SSN_PATTERN = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
SSN_LABEL = re.compile(r"(?i)\bssn\b|social\s+security|soc\.?\s*sec")
CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")  # Checked with Luhn below.
ACCOUNT_PATTERN = re.compile(r"(?i)\b(?:bank\s+)?(?:account|acct)\s*(?:no|number|#)?\.?\s*[:=#]\s*\d{6,17}\b")
ROUTING_PATTERN = re.compile(r"(?i)\b(?:routing|aba|rtn)\b[^\n\d]{0,20}(\d{9})(?!\d)")
IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")

CATEGORIES = ["Email", "Phone", "SSN", "Name", "Credit Card", "Date of Birth", "Address",
              "Passport", "Driver's License", "Bank Account", "IBAN", "Schedule"]
# A personal schedule: a dated, timed entry about a person's own appointment or whereabouts.
SCHEDULE_DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2})"
SCHEDULE_LINE = re.compile(
    r"(?im)^(?=.*" + SCHEDULE_DATE + r")(?=.*\b\d{1,2}(?::\d{2})\s*(?:am|pm)?\b)"
    r"(?=.*\b(?:appointment|doctor|dentist|therapy|therapist|clinic|pick ?up|drop ?off|visit|flight|interview|"
    r"meeting with|session|court|school run|surgery)\b).+$")
CALENDAR_EVENT = re.compile(r"BEGIN:VEVENT(?:(?!END:VEVENT).)*?DTSTART(?:(?!END:VEVENT).)*?(?:LOCATION|ATTENDEE)(?:(?!END:VEVENT).)*?END:VEVENT"
                            r"|BEGIN:VEVENT(?:(?!END:VEVENT).)*?(?:LOCATION|ATTENDEE)(?:(?!END:VEVENT).)*?DTSTART(?:(?!END:VEVENT).)*?END:VEVENT",
                            re.S)
HIGH_RISK = {"SSN", "Credit Card", "Passport", "Driver's License", "Bank Account", "IBAN"}

# CSV/TSV column names that make non-empty cells indicators even without a recognizable pattern.
COLUMN_CATEGORIES = {
    "name": "Name", "full_name": "Name", "employee_name": "Name", "first_name": "Name", "last_name": "Name",
    "email": "Email", "email_address": "Email", "phone": "Phone", "phone_number": "Phone", "mobile": "Phone",
    "ssn": "SSN", "social_security_number": "SSN", "dob": "Date of Birth", "date_of_birth": "Date of Birth",
    "birth_date": "Date of Birth", "address": "Address", "street_address": "Address",
    "credit_card": "Credit Card", "card_number": "Credit Card",
    "passport": "Passport", "passport_number": "Passport", "drivers_license": "Driver's License",
    "driver_license": "Driver's License", "license_number": "Driver's License",
    "account_number": "Bank Account", "bank_account": "Bank Account", "routing_number": "Bank Account",
    "iban": "IBAN", "appointment": "Schedule", "appointment_time": "Schedule",
}
TEXT_EXTENSIONS = {".txt", ".csv", ".tsv", ".log", ".md", ".json", ".xml", ".ini", ".cfg",
                   ".yaml", ".yml", ".gitignore", ".ics", ".vcf"}
OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
# Pre-2007 Office and Outlook files are OLE compound files; text is recovered as strings.
LEGACY_EXTENSIONS = {".doc", ".xls", ".ppt", ".msg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
# Formats this module can read. SUPPORTED_EXTENSIONS in config selects among them.
READABLE_EXTENSIONS = (TEXT_EXTENSIONS | OFFICE_EXTENSIONS | LEGACY_EXTENSIONS | {".zip", ".eml"}
                       | ({".pdf"} if pypdf else set()) | (IMAGE_EXTENSIONS if OCR else set()))
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
LEGACY_READ_FACTOR = 4  # Legacy files hold a lot of binary structure; read up to 4x the byte cap.
OCR_MAX_SIDE = 3000  # Larger images are scaled down before OCR (time and memory bound).
COUNT_CAP = 999  # Counts are reported without values and capped so a huge file stays cheap.
MAX_ARCHIVE_MEMBERS = 50  # Files examined inside one zip; nested archives are not opened.
MAX_PDF_PAGES = 50


# ---------- validators ----------
def luhn_valid(number):
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit = digit * 2 - 9 if digit * 2 > 9 else digit * 2
        total += digit
    return total % 10 == 0


def ssn_plausible(area, group, serial):
    """Structure the SSA never issues: area 000, 666 or 900-999; group 00; serial 0000."""
    area = int(area)
    return area != 0 and area != 666 and area < 900 and group != "00" and serial != "0000"


def aba_valid(number):
    digits = [int(c) for c in number]
    return len(digits) == 9 and sum(w * d for w, d in zip([3, 7, 1] * 3, digits)) % 10 == 0


def iban_valid(value):
    value = value.replace(" ", "")
    if not 15 <= len(value) <= 34:
        return False
    rearranged = value[4:] + value[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


def line_of(text, index):
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    return text[start:end if end != -1 else len(text)]


# ---------- decoding and readers ----------
def decode_bytes(raw, truncated=False):
    """Decode with BOM detection, a UTF-16 heuristic, strict UTF-8, then Windows-1252.

    `truncated` says the bytes were cut at the byte cap, so a broken final UTF-8
    character is expected and does not mean the file is in another encoding.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace"), "utf-16"
    sample = raw[:4096]
    if len(sample) >= 4:
        even_nuls, odd_nuls = sample[0::2].count(0), sample[1::2].count(0)
        half = len(sample) / 2
        if odd_nuls > 0.3 * half and even_nuls < 0.05 * half:
            return raw.decode("utf-16-le", errors="replace"), "utf-16-le"
        if even_nuls > 0.3 * half and odd_nuls < 0.05 * half:
            return raw.decode("utf-16-be", errors="replace"), "utf-16-be"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as error:
        if truncated and error.reason == "unexpected end of data":
            return raw.decode("utf-8", errors="replace"), "utf-8"
        return raw.decode("cp1252", errors="replace"), "cp1252"


def xml_text(data):
    """Text of an Office XML part; paragraph, row and cell ends become line breaks."""
    text = data.decode("utf-8", errors="replace")
    text = re.sub(r"</(?:w:p|a:p|row|si|c)>|<w:br/>|<w:tab/>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return (text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
                .replace("&apos;", "'").replace("&amp;", "&"))


def office_text(archive, budget):
    names = archive.namelist()
    parts = [n for n in names if n in ("word/document.xml", "xl/sharedStrings.xml")
             or re.fullmatch(r"(?:word/(?:header|footer)\d*|xl/worksheets/sheet\d+|ppt/slides/slide\d+|ppt/notesSlides/notesSlide\d+)\.xml", n)]
    chunks, used, truncated = [], 0, False
    for name in sorted(parts):
        info = archive.getinfo(name)
        if info.flag_bits & 0x1:
            raise ValueError("Encrypted document")
        with archive.open(info) as handle:
            data = handle.read(budget - used + 1)  # Stream: never trust the declared size.
        if len(data) > budget - used:
            data, truncated = data[:budget - used], True
        chunks.append(xml_text(data))
        used += len(data)
        if used >= budget:
            truncated = True
            break
    return "\n".join(chunks), truncated


def email_text(raw):
    message = email.message_from_bytes(raw, policy=policy.default)
    lines = [f"{header}: {message.get(header, '')}" for header in ("From", "To", "Cc", "Reply-To") if message.get(header)]
    for part in message.walk():
        if part.get_content_maintype() == "text":
            try:
                body = part.get_content()
            except (LookupError, ValueError):
                body = part.get_payload(decode=True) or b""
                body = decode_bytes(body)[0] if isinstance(body, bytes) else str(body)
            if part.get_content_subtype() == "html":
                body = re.sub(r"<[^>]+>", " ", body)
            lines.append(body)
    return "\n".join(lines)


def text_from_bytes(name, raw, budget, truncated=False):
    """(text, truncated, encoding) for one file's bytes, chosen by extension."""
    extension = extension_of(name)
    if extension in OFFICE_EXTENSIONS:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            text, truncated = office_text(archive, budget)
        return text, truncated, "office-xml"
    if extension == ".eml":
        return email_text(raw), False, "email"
    text, encoding = decode_bytes(raw, truncated)
    return text, False, encoding


def ocr_image(image):
    """Text from a Pillow image with Tesseract (bounded size and time)."""
    pytesseract, _ = OCR
    image = image.convert("L")
    if max(image.size) > OCR_MAX_SIDE:
        scale = OCR_MAX_SIDE / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return pytesseract.image_to_string(image, timeout=config.OCR_TIMEOUT_SECONDS)


def read_image(handle, budget):
    if not OCR:
        raise ValueError("OCR is not available")
    _, Image = OCR
    with Image.open(handle) as image:  # Pillow refuses decompression bombs above its pixel limit.
        text = ocr_image(image)
    return text[:budget], len(text) > budget


def read_pdf(handle, budget):
    """Page text; pages with almost no text (scans) are OCRed from their images when OCR is available."""
    reader = pypdf.PdfReader(handle)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF")
    chunks, used, images_read, ocr_used = [], 0, 0, False
    for page in reader.pages[:MAX_PDF_PAGES]:
        text = page.extract_text() or ""
        if len(text.strip()) < 20 and OCR and images_read < config.MAX_OCR_IMAGES:
            _, Image = OCR
            for embedded in page.images:
                if images_read >= config.MAX_OCR_IMAGES:
                    break
                try:
                    with Image.open(io.BytesIO(embedded.data)) as picture:
                        text += "\n" + ocr_image(picture)
                    ocr_used = True
                except Exception:  # One unreadable image does not fail the document.
                    pass
                images_read += 1
        chunks.append(text)
        used += len(text)
        if used >= budget:
            break
    text = "\n".join(chunks)
    return text[:budget], used > budget or len(reader.pages) > MAX_PDF_PAGES, ocr_used


def read_legacy(handle, budget):
    """Printable runs from an OLE file (.doc/.xls/.ppt/.msg): 8-bit text and UTF-16 text.

    This is the `strings` approach rather than a full parser: it recovers body text,
    cell strings and message properties well enough for pattern checks.
    """
    raw = handle.read(budget * LEGACY_READ_FACTOR + 1)
    if not raw.startswith(OLE_MAGIC):
        raise ValueError("Not an OLE compound file")
    truncated = len(raw) > budget * LEGACY_READ_FACTOR
    runs = [m.decode("cp1252") for m in re.findall(rb"[\x20-\x7e\x80-\xff\t\r\n]{4,}", raw)]
    runs += [m.decode("utf-16-le", errors="replace") for m in re.findall(rb"(?:[\x20-\x7e\t\r\n]\x00){4,}", raw)]
    text = "\n".join(runs).replace("\r", "\n")
    return text[:budget], truncated or len(text) > budget


def read_archive(handle, budget):
    """Text of readable members of a zip; nested archives and encrypted members are skipped."""
    chunks, used, members, truncated = [], 0, 0, False
    with zipfile.ZipFile(handle) as archive:
        for info in archive.infolist():
            if info.is_dir() or extension_of(info.filename) not in (TEXT_EXTENSIONS | OFFICE_EXTENSIONS | {".eml"}):
                continue
            if info.flag_bits & 0x1:
                continue
            if members >= MAX_ARCHIVE_MEMBERS or used >= budget:
                truncated = True
                break
            with archive.open(info) as member:
                raw = member.read(budget - used + 1)
            if len(raw) > budget - used:
                raw, truncated = raw[:budget - used], True
            try:
                text, cut, _ = text_from_bytes(info.filename, raw, budget - used, truncated)
            except (zipfile.BadZipFile, ValueError, KeyError):
                continue
            truncated = truncated or cut
            chunks.append(text)
            used += len(raw)
            members += 1
    return "\n".join(chunks), truncated, members


# ---------- counting ----------
def extension_of(path):
    path = Path(path)
    return ".gitignore" if path.name.lower() == ".gitignore" else path.suffix.lower()


def supported(path):
    extension = extension_of(path)
    return extension in config.SUPPORTED_EXTENSIONS and extension in READABLE_EXTENSIONS


def count_indicators(text):
    """Return {category: occurrences}; matched values are counted and then discarded."""
    counts = {name: len(re.findall(pattern, text)) for name, pattern in PATTERNS.items()}
    counts["Phone"] = len(US_PHONE.findall(text)) + sum(
        8 <= sum(c.isdigit() for c in m.group()) <= 15 for m in INTL_PHONE.finditer(text))
    # An impossible SSN structure (e.g. 000-12-3456) counts only when the line labels it as an SSN.
    counts["SSN"] = sum(ssn_plausible(*m.groups()) or bool(SSN_LABEL.search(line_of(text, m.start())))
                        for m in SSN_PATTERN.finditer(text))
    counts["Credit Card"] = sum(luhn_valid(m.group()) for m in CARD_CANDIDATE.finditer(text))
    counts["Bank Account"] = len(ACCOUNT_PATTERN.findall(text)) + sum(aba_valid(n) for n in ROUTING_PATTERN.findall(text))
    counts["IBAN"] = sum(iban_valid(m.group()) for m in IBAN_PATTERN.finditer(text))
    counts["Schedule"] = len(SCHEDULE_LINE.findall(text)) + len(CALENDAR_EVENT.findall(text))
    return {name: min(COUNT_CAP, counts.get(name, 0)) for name in CATEGORIES}


def count_table_columns(text, delimiter, counts):
    """Non-empty cells under recognizable column headers count as indicators."""
    columns = dict.fromkeys(counts, 0)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        for field, value in row.items():
            if not field or not isinstance(value, str) or not value.strip():
                continue
            category = COLUMN_CATEGORIES.get(re.sub(r"[\s'-]+", "_", field.strip().lower()))
            if category:
                columns[category] += 1
    for category, cells in columns.items():  # max(), not a sum: one value must not count twice.
        counts[category] = min(COUNT_CAP, max(counts[category], cells))


def severity(found):
    """Plain-language band for the verified indicators (not a measure of identity)."""
    if not found:
        return "None"
    if HIGH_RISK & set(found) or len(found) >= 3:
        return "High"
    return "Medium" if len(found) == 2 else "Low"


def inspect_file(path, root):
    result = {"status": "Inspection unsupported", "indicators": {}}
    if not supported(path):
        return result  # Unsupported files are never opened.
    try:
        extension = extension_of(path)
        budget = config.MAX_INSPECTION_BYTES
        # Revalidate and open without following links; every reader uses this one handle.
        with safe_open(path, root) as handle:
            if extension == ".pdf":
                text, truncated, ocr_used = read_pdf(handle, budget)
                result["encoding"] = "pdf+ocr" if ocr_used else "pdf"
            elif extension == ".zip":
                text, truncated, members = read_archive(handle, budget)
                result.update(encoding="zip", archive_members=members)
            elif extension in OFFICE_EXTENSIONS:
                with zipfile.ZipFile(handle) as archive:  # Seeks on disk; parts are streamed with a cap.
                    text, truncated = office_text(archive, budget)
                result["encoding"] = "office-xml"
            elif extension in LEGACY_EXTENSIONS:
                text, truncated = read_legacy(handle, budget)
                result["encoding"] = "ole-strings"
            elif extension in IMAGE_EXTENSIONS:
                text, truncated = read_image(handle, budget)
                result["encoding"] = "ocr"
            else:
                raw = handle.read(budget + 1)
                text, cut, encoding = text_from_bytes(path, raw[:budget], budget, len(raw) > budget)
                truncated = len(raw) > budget or cut
                result["encoding"] = encoding
        result["truncated"] = truncated
        if Path(path).name.lower() == ".gitignore":
            clue = bool(re.search(r"(?i)payroll|personnel|contacts|private|employee", text))
            result.update(status="Metadata clues detected" if clue else "No supported metadata clues detected",
                          indicators={"Sensitive pattern metadata clue (not proof of PII)": clue},
                          counts={}, severity="Low" if clue else "None")
            return result
        counts = count_indicators(text)
        if extension in (".csv", ".tsv"):
            count_table_columns(text, "\t" if extension == ".tsv" else ",", counts)
        flags = {name: counts[name] > 0 for name in CATEGORIES}
        found = [name for name, hit in flags.items() if hit]
        result.update(status="PII INDICATORS DETECTED" if found else "No supported PII indicators detected",
                      indicators=flags, counts={name: counts[name] for name in found},
                      severity=severity(found))
    except Exception:  # Any reader failure is reported per file, never fatal to a scan.
        result["status"] = "Inspection failed"
    return result


def inspect_candidates(candidates, approved):
    if not approved:
        return []
    results = []
    successful = 0
    for candidate in candidates:
        if successful >= config.TOP_K_FILES:
            break
        if not supported(candidate["path"]):
            continue
        result = inspect_file(candidate["path"], candidate["root"])
        results.append((candidate, result))
        if result["status"] != "Inspection failed":
            successful += 1
    return results
