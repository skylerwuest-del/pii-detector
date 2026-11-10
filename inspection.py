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
    "Name": r"(?im)^\s*(?:name|full[ _]name|employee[ _]name)\s*:\s*\S+",
}


def supported(path):
    path = Path(path)
    extension = ".gitignore" if path.name.lower() == ".gitignore" else path.suffix.lower()
    return extension in config.SUPPORTED_EXTENSIONS and extension in (".txt", ".csv", ".gitignore")


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
                          indicators={"Sensitive pattern metadata clue (not proof of PII)": clue})
            return result
        flags = {name: bool(re.search(pattern, text)) for name, pattern in PATTERNS.items()}
        if checked.suffix.lower() == ".csv":
            for row in csv.DictReader(io.StringIO(text)):
                for field, value in row.items():
                    if field and field.strip().lower() in ("name", "full_name", "employee_name") and value and str(value).strip():
                        flags["Name"] = True
        result.update(status="PII INDICATORS DETECTED" if any(flags.values()) else "No supported PII indicators detected",
                      indicators=flags)
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
