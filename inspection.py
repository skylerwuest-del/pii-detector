"""Deterministic verification rules, not another ML algorithm. Never return matched values."""
import csv
import io
import re
from pathlib import Path
import config
from scanner import safe_path

PATTERNS = {
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "Phone": r"(?<!\d)(?:\+1[ -]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)",
    "SSN": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "Name": r"(?im)^\s*(?:name|full[ _]name|employee[ _]name|first[ _]name|last[ _]name)\s*:\s*\S+",
    "Date of Birth": r"(?i)\b(?:dob|date[ _]of[ _]birth|birth[ _]?date|birthday)\b\s*[:=]\s*\S+",
    "Address": r"(?i)\b\d{1,5}\s+(?:[A-Za-z0-9.'-]+\s+){1,4}(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)\b\.?",
}
CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")  # Checked with Luhn below.
CATEGORIES = list(PATTERNS) + ["Credit Card"]
CATEGORIES.insert(CATEGORIES.index("Name") + 1, CATEGORIES.pop())  # Keep the report order stable.
HIGH_RISK = {"SSN", "Credit Card"}

# CSV/TSV column names that make non-empty cells indicators even without a recognizable pattern.
COLUMN_CATEGORIES = {
    "name": "Name", "full_name": "Name", "employee_name": "Name", "first_name": "Name", "last_name": "Name",
    "email": "Email", "email_address": "Email", "phone": "Phone", "phone_number": "Phone", "mobile": "Phone",
    "ssn": "SSN", "social_security_number": "SSN", "dob": "Date of Birth", "date_of_birth": "Date of Birth",
    "birth_date": "Date of Birth", "address": "Address", "street_address": "Address",
    "credit_card": "Credit Card", "card_number": "Credit Card",
}
# Formats that can be read as UTF-8 text. SUPPORTED_EXTENSIONS in config selects among them.
READABLE_EXTENSIONS = {".txt", ".csv", ".tsv", ".log", ".md", ".json", ".xml", ".ini", ".cfg",
                       ".yaml", ".yml", ".gitignore"}
COUNT_CAP = 999  # Counts are reported without values and capped so a huge file stays cheap.


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


def extension_of(path):
    path = Path(path)
    return ".gitignore" if path.name.lower() == ".gitignore" else path.suffix.lower()


def supported(path):
    extension = extension_of(path)
    return extension in config.SUPPORTED_EXTENSIONS and extension in READABLE_EXTENSIONS


def count_indicators(text):
    """Return {category: occurrences}; matched values are counted and then discarded."""
    counts = {name: min(COUNT_CAP, len(re.findall(pattern, text))) for name, pattern in PATTERNS.items()}
    counts["Credit Card"] = min(COUNT_CAP, sum(luhn_valid(m.group()) for m in CARD_CANDIDATE.finditer(text)))
    return counts


def count_table_columns(text, delimiter, counts):
    """Non-empty cells under recognizable column headers count as indicators."""
    columns = dict.fromkeys(counts, 0)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        for field, value in row.items():
            if not field or not isinstance(value, str) or not value.strip():
                continue
            category = COLUMN_CATEGORIES.get(re.sub(r"[\s-]+", "_", field.strip().lower()))
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
        checked = safe_path(path, root)  # Revalidate just before reading.
        with checked.open("rb") as handle:
            raw = handle.read(config.MAX_INSPECTION_BYTES + 1)
        result["truncated"] = len(raw) > config.MAX_INSPECTION_BYTES
        text = raw[:config.MAX_INSPECTION_BYTES].decode("utf-8", errors="replace")
        if checked.name.lower() == ".gitignore":
            clue = bool(re.search(r"(?i)payroll|personnel|contacts|private|employee", text))
            result.update(status="Metadata clues detected" if clue else "No supported metadata clues detected",
                          indicators={"Sensitive pattern metadata clue (not proof of PII)": clue},
                          counts={}, severity="Low" if clue else "None")
            return result
        counts = count_indicators(text)
        if checked.suffix.lower() in (".csv", ".tsv"):
            count_table_columns(text, "\t" if checked.suffix.lower() == ".tsv" else ",", counts)
        flags = {name: counts[name] > 0 for name in CATEGORIES}
        found = [name for name, hit in flags.items() if hit]
        result.update(status="PII INDICATORS DETECTED" if found else "No supported PII indicators detected",
                      indicators=flags, counts={name: counts[name] for name in found},
                      severity=severity(found))
    except (OSError, ValueError, csv.Error):
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
