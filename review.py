"""Step through verified files and decide what to do with each one.

Nothing is ever permanently deleted here. "Quarantine" moves a file into
outputs/quarantine (restorable with `restore`); "recycle" sends it to the
Recycle Bin / Trash, which the operating system can undo. Every decision can be
saved as a training label (data/feedback.json) so the CNN learns your computer.
"""
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import config
import ui
from scanner import is_link

ACTIONS = "[o]pen  [k]eep (is PII)  [n]ot PII  [q]uarantine  [r]ecycle  [s]kip  [x] stop"


# ---------- feedback labels ----------
def feedback_file():
    return config.DATA_DIRECTORY / "feedback.json"


def load_feedback():
    try:
        return json.loads(feedback_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def record_feedback(path, root, label, size=None):
    """Save (relative path, label). The relative path is what the CNN sees; it stays on this computer."""
    try:
        relative = Path(path).relative_to(root).as_posix()
    except ValueError:
        relative = Path(path).name
    entries = [e for e in load_feedback() if e.get("path") != relative]
    entries.append(dict(path=relative, label=int(label), size=size, time=time.time()))
    feedback_file().parent.mkdir(parents=True, exist_ok=True)
    feedback_file().write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return relative


# ---------- file actions ----------
def check_regular_file(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        raise ValueError(f"{path} no longer exists")
    if is_link(path) or not path.is_file():
        raise ValueError("Not a regular file (missing, moved, or a link)")
    return path


def open_file(path):
    """Open with the default application. Arguments are passed as a list, never through a shell."""
    path = check_regular_file(path)
    if os.name == "nt":
        os.startfile(str(path))  # noqa: S606 - opens the user's own file in its default app
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def manifest_file():
    return Path(config.QUARANTINE_DIRECTORY) / "manifest.json"


def load_manifest():
    try:
        return json.loads(manifest_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def quarantine(path):
    """Move a file into the quarantine folder and remember where it came from."""
    path = check_regular_file(path)
    entry_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + f"{time.monotonic_ns() % 100000:05d}"
    target_folder = Path(config.QUARANTINE_DIRECTORY) / entry_id
    target_folder.mkdir(parents=True, exist_ok=False)
    target = target_folder / path.name
    shutil.move(str(path), str(target))
    entries = load_manifest()
    entries.append(dict(id=entry_id, original=str(path), stored=str(target), time=time.time(), restored=False))
    manifest_file().write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return target


def restore(entry_id):
    entries = load_manifest()
    for entry in entries:
        if entry["id"] == entry_id and not entry.get("restored"):
            original, stored = Path(entry["original"]), Path(entry["stored"])
            if original.exists():
                raise ValueError(f"Something already exists at {original}; move it first")
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stored), str(original))
            entry["restored"] = True
            manifest_file().write_text(json.dumps(entries, indent=2), encoding="utf-8")
            try:
                stored.parent.rmdir()
            except OSError:
                pass
            return original
    raise ValueError(f"No quarantined file with id {entry_id}")


def send_to_trash(path):
    """Recycle Bin on Windows, Trash elsewhere. Both can be undone by the operating system."""
    path = check_regular_file(path).absolute()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
        # FO_DELETE with FOF_ALLOWUNDO (Recycle Bin), no confirmation/progress/error UI.
        operation = SHFILEOPSTRUCTW(None, 3, str(path) + "\0", None, 0x40 | 0x10 | 0x4 | 0x400, False, None, None)
        if ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation)) or operation.fAnyOperationsAborted:
            raise OSError("The Recycle Bin operation failed")
        return "Recycle Bin"
    if sys.platform == "darwin":
        trash = Path.home() / ".Trash"
        target = trash / path.name
        counter = 1
        while target.exists():
            target = trash / f"{path.stem} ({counter}){path.suffix}"
            counter += 1
        shutil.move(str(path), str(target))
        return str(trash)
    # freedesktop.org Trash specification: files/ plus an info/*.trashinfo record for restore.
    trash = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "Trash"
    (trash / "files").mkdir(parents=True, exist_ok=True)
    (trash / "info").mkdir(parents=True, exist_ok=True)
    name, counter = path.name, 1
    while (trash / "files" / name).exists() or (trash / "info" / f"{name}.trashinfo").exists():
        name = f"{path.stem}.{counter}{path.suffix}"
        counter += 1
    from urllib.parse import quote
    (trash / "info" / f"{name}.trashinfo").write_text(
        f"[Trash Info]\nPath={quote(str(path))}\nDeletionDate={datetime.datetime.now():%Y-%m-%dT%H:%M:%S}\n", encoding="utf-8")
    shutil.move(str(path), str(trash / "files" / name))
    return str(trash)


# ---------- interactive review ----------
def load_review_items(include_clean=False):
    report = config.OUTPUT_DIRECTORY / "inspection_report.csv"
    try:
        with report.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    items = [r for r in rows if include_clean or r.get("indicator_counts")]
    return [r for r in items if Path(r["absolute_path"]).is_file()]


def confirm(ask, question):
    return ask(f"{question} [y/N]: ").strip().lower() in ("y", "yes")


def review(include_clean=False, ask=input):
    """Interactive loop. Returns {action: count}. `ask` is injectable for tests."""
    items = load_review_items(include_clean)
    summary = {"kept": 0, "dismissed": 0, "quarantined": 0, "recycled": 0, "skipped": 0}
    if not items:
        ui.info("Nothing to review. Scan and inspect a folder first (only files that still exist are listed).")
        return summary
    ui.heading(f"Review {len(items)} file(s)")
    print("Decisions are saved as training labels (data/feedback.json); retrain to use them.")
    for number, item in enumerate(items, 1):
        path, root = item["absolute_path"], item.get("scan_root") or str(Path(item["absolute_path"]).parent)
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = None
        print()
        print(ui.style(f"[{number}/{len(items)}] ", "bold") + path)
        counts = item.get("indicator_counts", "").replace(";", ", ").replace("=", " x") or "no indicators"
        print(f"      severity {item.get('severity', '?')}  |  {counts}  |  CNN score {item.get('cnn_score')}"
              + (f"  |  {ui.human_size(size)}" if size is not None else ""))
        while True:
            choice = ask(ACTIONS + "\n> ").strip().lower()[:1]
            try:
                if choice == "o":
                    open_file(path)
                    continue
                if choice == "k":
                    record_feedback(path, root, 1, size)
                    summary["kept"] += 1
                elif choice == "n":
                    record_feedback(path, root, 0, size)
                    summary["dismissed"] += 1
                    ui.ok("Marked as not PII.")
                elif choice == "q":
                    if not confirm(ask, "Move this file into the quarantine folder?"):
                        continue
                    target = quarantine(path)
                    record_feedback(path, root, 1, size)
                    summary["quarantined"] += 1
                    ui.ok(f"Quarantined to {target} (restore with the `restore` command).")
                elif choice == "r":
                    if not confirm(ask, "Send this file to the Recycle Bin / Trash?"):
                        continue
                    where = send_to_trash(path)
                    record_feedback(path, root, 1, size)
                    summary["recycled"] += 1
                    ui.ok(f"Moved to {where}; it can be restored from there.")
                elif choice == "s":
                    summary["skipped"] += 1
                elif choice == "x":
                    return finish(summary)
                else:
                    ui.warn("Choose one of: " + ACTIONS)
                    continue
                break
            except (OSError, ValueError) as error:
                ui.error(f"{error}")
                break
    return finish(summary)


def finish(summary):
    print()
    ui.info(", ".join(f"{n} {k}" for k, n in summary.items()))
    if summary["kept"] + summary["dismissed"] + summary["quarantined"] + summary["recycled"]:
        print(f"{len(load_feedback())} feedback labels saved. Retrain (menu option 3) to teach the model your decisions.")
    return summary


def restore_interactive(ask=input):
    entries = [e for e in load_manifest() if not e.get("restored")]
    if not entries:
        ui.info("The quarantine folder is empty.")
        return
    ui.table(["#", "Quarantined", "Original location"],
             [[i, datetime.datetime.fromtimestamp(e["time"]).strftime("%Y-%m-%d %H:%M"), e["original"]] for i, e in enumerate(entries, 1)],
             aligns=[">", "<", "<"])
    choice = ask("Number to restore (Enter to cancel): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(entries):
        ui.ok(f"Restored {restore(entries[int(choice) - 1]['id'])}")
