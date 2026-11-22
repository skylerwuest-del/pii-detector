"""Terminal menu, navigator shell and command-line interface.

Broad scans always separate metadata and content approval. Run with no arguments
for the menu, or e.g. `python main.py scan "C:\\Users\\me\\Documents" --html`.
"""
import argparse
import datetime
import json
import os
import shlex
import sys
import webbrowser
import config
import ui
from scanner import Navigator, fixed_drives, common_places
from inference import (scan, print_inspections, print_rows, print_folders, query_results,
                       read_meta, database_for, interrupted_scan, RankedCandidates)

COMMANDS = [
    ("ls", "List this folder (folders first, with sizes)"),
    ("cd <folder>", "Change folder; 'cd ..' up, 'cd ~' home, 'cd -' previous"),
    ("pwd", "Show the current folder"),
    ("tree [depth]", "Show a bounded tree of this folder (default depth 2)"),
    ("places", "Jump list: Documents, Downloads, Desktop ..."),
    ("select", "Lock THIS folder as the authorized scan root"),
    ("scan", "Rank every file under the locked root, then verify the top ones"),
    ("results [filters]", "Ranked list: --high --ext csv --folder Documents --changed"),
    ("next / prev", "Page through the last results list"),
    ("folders [depth]", "Riskiest folders under the root"),
    ("inspect", "Verify the top readable files of the last scan again"),
    ("review [--all]", "Go through verified files: open, keep, dismiss, quarantine, recycle"),
    ("restore", "Put a quarantined file back"),
    ("report", "Write an HTML report of the last scan and open it"),
    ("help", "Show this list"),
    ("back", "Return to the main menu"),
    ("exit", "Quit"),
]
HELP = " | ".join(name.split()[0] for name, _ in COMMANDS)
MAX_LISTED = 200


def print_help():
    ui.table(["Command", "What it does"], [[ui.style(n, "cyan"), d] for n, d in COMMANDS])


def print_listing(nav):
    entries = nav.listing()
    for name, folder, size, linked in entries[:MAX_LISTED]:
        if folder:
            print(ui.style(f"{name}/", "blue", "bold") + (ui.style("  [link, not followed]", "gray") if linked else ""))
        else:
            print(f"{name}  " + ui.style(ui.human_size(size), "gray") + (ui.style("  [link, not followed]", "gray") if linked else ""))
    if len(entries) > MAX_LISTED:
        print(ui.style(f"... and {len(entries) - MAX_LISTED} more", "gray"))
    folders = sum(1 for e in entries if e[1])
    print(ui.style(f"{folders} folders, {len(entries) - folders} files", "gray"))


def enable_completion(nav):
    """Tab-complete commands and sub-folders where the platform provides readline."""
    try:
        import readline
    except ImportError:
        return
    names = [name.split()[0] for name, _ in COMMANDS]
    def complete(text, state):
        line = readline.get_line_buffer().lstrip()
        if " " not in line:
            options = [n + " " for n in names if n.startswith(text)]
        else:
            try:
                options = [name + "/" for name, folder, _, _ in nav.listing() if folder and name.startswith(text)]
            except OSError:
                options = []
        return options[state] if state < len(options) else None
    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")


def prompt_for(nav):
    lock = ui.style(" [root locked]", "green") if nav.root else ""
    return f"{ui.style(ui.shorten(nav.current, 48), 'cyan')}{lock}> "


def parse_filters(argument):
    """'--high --ext csv --folder "My Docs" --changed' -> dict for query_results."""
    parser = argparse.ArgumentParser(prog="results", add_help=False, exit_on_error=False)
    parser.add_argument("--high", action="store_true")
    parser.add_argument("--ext")
    parser.add_argument("--folder")
    parser.add_argument("--changed", action="store_true")
    try:
        words = shlex.split(argument, posix=os.name != "nt")
        args, unknown = parser.parse_known_args([w.strip('"') for w in words])
    except (ValueError, argparse.ArgumentError) as error:
        raise ValueError(f"Could not read filters: {error}")
    if unknown:
        raise ValueError(f"Unknown filter(s): {' '.join(unknown)}. Use --high --ext X --folder X --changed")
    return dict(high=args.high, extension=args.ext, folder=args.folder, changed=args.changed)


class ResultsPager:
    """Remembers the last filters so `next` / `prev` page through them."""
    def __init__(self):
        self.filters, self.offset, self.full = {}, 0, False

    def show(self, filters=None, step=0, full=None):
        if full is not None:
            self.full = full
        database = database_for(self.full)
        if not database.is_file():
            raise ValueError("No scan results yet. Run scan first.")
        if filters is not None:
            self.filters, self.offset = filters, 0
        size = config.RESULTS_PAGE_SIZE
        if step > 0 and self.offset + size >= query_results(database, limit=0, **self.filters)[1]:
            ui.info("Already on the last page.")
            return
        if step < 0 and self.offset == 0:
            ui.info("Already on the first page.")
            return
        self.offset = max(0, self.offset + step * size)
        rows, total, threshold = query_results(database, offset=self.offset, limit=size, **self.filters)
        if not rows and self.offset and total:
            self.offset = max(0, (total - 1) // size * size)
            rows, total, threshold = query_results(database, offset=self.offset, limit=size, **self.filters)
        active = ", ".join(k if v is True else f"{k}={v}" for k, v in self.filters.items() if v) or "none"
        ui.heading(f"Results {self.offset + 1 if rows else 0}-{self.offset + len(rows)} of {total:,}  (filters: {active})")
        if rows:
            print_rows(rows, threshold, self.offset, multi_root=len(read_meta(database).get("roots", [])) > 1)
            if self.offset + len(rows) < total:
                print(ui.style("Type next for more, prev to go back.", "gray"))
        else:
            print("No files match these filters.")


def open_report(full=False):
    from report import write_html_report
    path = write_html_report(full=full)
    ui.ok(f"Report written: {path}")
    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception:  # A missing browser is not an error for the report itself.
        pass
    return path


def navigate():
    nav = Navigator()
    pager = ResultsPager()
    enable_completion(nav)
    ui.heading("Controlled directory scan")
    print("Browse with ls / cd, then `select` the folder you are authorized to audit and `scan` it.")
    print("Select only a synthetic test directory or a directory you are authorized to audit. Type help for commands.")
    while True:
        try:
            command = input(prompt_for(nav)).strip()
            action, _, argument = command.partition(" ")
            argument = argument.strip()
            if not action:
                continue
            if action in ("ls", "dir"):
                print_listing(nav)
            elif action == "cd":
                nav.cd(argument.strip('"'))
            elif action == "pwd":
                print(nav.current)
            elif action == "tree":
                print(nav.current)
                print("\n".join(nav.tree_lines(int(argument) if argument.isdigit() else 2)))
            elif action == "places":
                places = common_places()
                for place in places:
                    print(place)
                if not places:
                    print("No standard folders found under your home directory.")
            elif action == "select":
                nav.select()
                ui.ok(f"Locked root: {nav.root}")
                print("Navigation is now confined to this folder. Type `scan` to rank its files.")
            elif action == "scan":
                if nav.root is None:
                    ui.warn("Use select to lock an authorized scan root first.")
                else:
                    ui.info(f"Scanning locked root {nav.root}  (Ctrl+C cancels; a rescan reuses finished scores)")
                    print_inspections(scan([nav.root]), approved=True)
                    print(ui.style("Next: review | results --high | folders | report", "gray"))
            elif action == "results":
                pager.show(parse_filters(argument), full=False)
            elif action == "next":
                pager.show(step=1)
            elif action == "prev":
                pager.show(step=-1)
            elif action == "folders":
                if not database_for(False).is_file():
                    raise ValueError("No scan results yet. Run scan first.")
                ui.heading("Riskiest folders")
                print_folders(database_for(False), int(argument) if argument.isdigit() else 1, limit=15)
            elif action == "inspect":
                database = database_for(False)
                if nav.root is None or not database.is_file() or str(nav.root) not in read_meta(database).get("roots", []):
                    raise ValueError("Select and scan this root first.")
                print_inspections(RankedCandidates(database), approved=True)
            elif action == "review":
                from review import review
                review(include_clean=argument == "--all")
            elif action == "restore":
                from review import restore_interactive
                restore_interactive()
            elif action == "report":
                open_report(full=False)
            elif action == "help":
                print_help()
            elif action in ("back", "b"):
                return
            elif action in ("exit", "quit", "q"):
                raise SystemExit
            else:
                ui.warn(f"Unknown command '{action}'. Type help for the list.")
        except (OSError, ValueError) as error:
            ui.error(f"Operation failed: {error}")
        except KeyboardInterrupt:
            print()
            ui.info("Use back to return or exit to quit.")
        except EOFError:
            return


def full_mode():
    drives = fixed_drives()
    ui.heading("Full computer audit")
    print("Detected local fixed drives:", ", ".join(map(str, drives)) or "none")
    ui.warn("Only proceed on a computer you own or are explicitly authorized to audit. Ctrl+C cancels.")
    print("System folders are skipped; no permissions are bypassed. Stage 1 reads file NAMES only.")
    previous = interrupted_scan(full=True)
    if previous:
        print(f"A previous full scan stopped after {previous.get('files', 0):,} files; finished files keep their scores and are not re-scored.")
    if not drives:
        return
    if config.FULL_SCAN_REQUIRE_CONFIRMATION and input("Type SCAN to approve metadata scanning: ").strip() != "SCAN":
        ui.info("Cancelled: nothing was scanned.")
        return
    candidates = scan(drives, full=True)
    if config.FULL_SCAN_ALLOW_CONTENT_INSPECTION:
        print(f"Stage 2 would open ONLY the top {config.TOP_K_FILES} readable files to look for PII patterns.")
        approved = input("Type INSPECT to approve reading only the top supported candidates: ").strip() == "INSPECT"
        print_inspections(candidates, approved)
        if approved:
            print(ui.style("Next: menu option 5 reviews these files.", "gray"))


def model_status():
    """(trained?, one-line summary) for the menu header."""
    checkpoint = config.OUTPUT_DIRECTORY / "model.pt"
    if not checkpoint.is_file():
        return False, ui.style("No trained model yet", "yellow") + " (choose 3, or you will be offered training before a scan)"
    when = datetime.datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    detail = ""
    try:
        m = json.loads((config.OUTPUT_DIRECTORY / "training_metrics.json").read_text(encoding="utf-8"))
        detail = f"  |  test F1 {m['f1']:.3f}  recall {m['recall']:.3f}  AUC {m.get('auc', float('nan')):.3f}"
    except (OSError, ValueError, KeyError):
        pass
    return True, ui.style("Model ready", "green") + f" (trained {when}){detail}"


def show_model_details():
    ui.heading("Model details")
    try:
        m = json.loads((config.OUTPUT_DIRECTORY / "training_metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ui.warn("No training report found. Train the model first.")
        return
    ui.table(["Item", "Value"], [
        ["Precision / recall / F1", f"{m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}"],
        ["ROC AUC", f"{m.get('auc', float('nan')):.3f}"],
        ["Decision threshold", f"{m.get('threshold', config.PREDICTION_THRESHOLD):.2f}"],
        ["Calibration error (test)", f"{m['calibration_error']:.3f} (temperature {m.get('temperature', 1):.2f})" if "calibration_error" in m else "n/a"],
        ["Feedback labels used", m.get("feedback_examples", 0)],
        ["Metadata features", "on" if m.get("metadata_features") else "off"],
        ["Test examples", m.get("test_examples", "?")],
        ["Epochs run / best epoch", f"{m.get('epochs_run', len(m['history']))} / {m['best_epoch']}"],
        ["Seed", m["seed"]],
        ["Training time", ui.human_time(m.get("seconds", 0))],
    ], aligns=["<", ">"])
    print("\nSynthetic-data results do not prove real-world accuracy. Retrain after changing config.py model settings.")


def train_model(ask=input):
    from generate_data import generate
    from train import train
    checkpoint = config.OUTPUT_DIRECTORY / "model.pt"
    if checkpoint.is_file() and ask("This replaces the saved checkpoint. Retrain? [y/N]: ").strip().lower() != "y":
        ui.info("Cancelled: the existing model was kept.")
        return
    ui.heading("Generating synthetic data")
    generate()
    train()


def ensure_model():
    """Offer training instead of failing when a scan is requested without a checkpoint."""
    if (config.OUTPUT_DIRECTORY / "model.pt").is_file():
        return True
    ui.warn("No trained model found.")
    if input("Generate synthetic data and train now? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        train_model()
    return (config.OUTPUT_DIRECTORY / "model.pt").is_file()


# ---------- settings ----------
def parse_setting(name, text):
    """Validate a typed value against the kind of setting it is."""
    text = text.strip()
    default = config.DEFAULTS[name]
    if name == "MAX_FILES_TO_SCAN":
        if text.lower() in ("none", "", "no limit"):
            return None
        value = int(text)
        if value < 0:
            raise ValueError("must be 0 or more, or None")
        return value
    if isinstance(default, bool):
        if text.lower() in ("true", "yes", "on", "1"):
            return True
        if text.lower() in ("false", "no", "off", "0"):
            return False
        raise ValueError("answer true or false")
    if isinstance(default, float):
        value = float(text)
        if not 0.0 < value < 1.0:
            raise ValueError("must be between 0 and 1")
        return value
    value = int(text)
    minimum = 1024 if name == "MAX_INSPECTION_BYTES" else 0
    if value < minimum:
        raise ValueError(f"must be at least {minimum}")
    return value


def save_settings():
    changed = {n: getattr(config, n) for n in config.USER_SETTABLE if getattr(config, n) != config.DEFAULTS[n]}
    if changed:
        config.USER_SETTINGS_FILE.write_text(json.dumps(changed, indent=2), encoding="utf-8")
    elif config.USER_SETTINGS_FILE.exists():
        config.USER_SETTINGS_FILE.unlink()
    return changed


def settings_menu(ask=input):
    while True:
        ui.heading("Settings")
        rows = [[i, name, repr(getattr(config, name)), "" if getattr(config, name) == config.DEFAULTS[name] else ui.style("changed", "yellow")]
                for i, name in enumerate(config.USER_SETTABLE, 1)]
        ui.table(["#", "Setting", "Value", ""], rows, aligns=[">", "<", "<", "<"])
        print("\nEnter a number to change it, s to save, r to reset all to config.py defaults, Enter to go back.")
        print("Model and training settings stay in config.py (retrain after editing them).")
        choice = ask("Settings> ").strip().lower()
        if not choice:
            return
        if choice == "s":
            changed = save_settings()
            ui.ok(f"Saved {len(changed)} change(s) to {config.USER_SETTINGS_FILE}" if changed else "No changes from defaults; nothing saved.")
        elif choice == "r":
            for name, value in config.DEFAULTS.items():
                setattr(config, name, value)
            save_settings()
            ui.ok("Reset to config.py defaults.")
        elif choice.isdigit() and 1 <= int(choice) <= len(config.USER_SETTABLE):
            name = config.USER_SETTABLE[int(choice) - 1]
            try:
                setattr(config, name, parse_setting(name, ask(f"New value for {name} (now {getattr(config, name)!r}): ")))
                ui.ok(f"{name} = {getattr(config, name)!r} for this session (s saves it).")
            except ValueError as error:
                ui.error(f"Not changed: {error}")
        else:
            ui.warn("Choose a number, s, r, or Enter.")


def main():
    while True:
        ui.banner("CS471 - PII DETECTION AGENT", "Character-level CNN ranks file paths; deterministic rules verify the top files")
        print(model_status()[1], "\n")
        print("  [1] Controlled directory scan   - pick a folder you are authorized to audit")
        print("  [2] Full computer scan          - all fixed drives (separate approvals)")
        print("  [3] Train / retrain the model")
        print("  [4] Model details")
        print("  [5] Review flagged files        - open, keep, dismiss, quarantine or recycle")
        print("  [6] Restore from quarantine")
        print("  [7] Settings")
        print("  [0] Exit\n")
        try:
            selection = input("Enter selection: ").strip().lower() or str(config.DEFAULT_SCAN_MODE)
            if selection == "1" and ensure_model():
                navigate()
            elif selection == "2" and ensure_model():
                full_mode()
            elif selection == "3":
                train_model()
            elif selection == "4":
                show_model_details()
            elif selection == "5":
                from review import review
                review()
            elif selection == "6":
                from review import restore_interactive
                restore_interactive()
            elif selection == "7":
                settings_menu()
            elif selection in ("0", "q", "quit", "exit"):
                return
            elif selection not in ("1", "2"):
                ui.warn(f"'{selection}' is not a menu option.")
        except (OSError, ValueError) as error:
            ui.error(f"Unable to continue: {error}")
        except (EOFError, KeyboardInterrupt):
            return
        print()


# ---------- command line ----------
def build_parser():
    parser = argparse.ArgumentParser(prog="main.py", description="PII Detection Agent. No arguments opens the menu.")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("scan", help="scan one folder you are authorized to audit (passing it is the authorization)")
    p.add_argument("folder")
    p.add_argument("--top", type=int, help="files to verify (TOP_K_FILES)")
    p.add_argument("--no-inspect", action="store_true", help="rank names only; open no file contents")
    p.add_argument("--html", action="store_true", help="also write and open the HTML report")
    p.add_argument("--fail-on-pii", action="store_true", help="exit code 3 when any verified file has PII indicators")
    p = sub.add_parser("full", help="audit all fixed drives (Windows)")
    p.add_argument("--yes", action="store_true", help="skip typing SCAN (authorized automation only)")
    p.add_argument("--inspect", action="store_true", help="also verify the top files without the INSPECT prompt")
    p.add_argument("--top", type=int, help="files to verify (TOP_K_FILES)")
    p.add_argument("--html", action="store_true", help="also write the HTML report")
    p = sub.add_parser("results", help="show ranked results of the last scan")
    p.add_argument("--full", action="store_true", help="the last full-computer scan instead of the last folder scan")
    p.add_argument("--high", action="store_true")
    p.add_argument("--ext")
    p.add_argument("--folder")
    p.add_argument("--changed", action="store_true")
    p.add_argument("--limit", type=int, default=50)
    p = sub.add_parser("report", help="write the HTML report of the last scan")
    p.add_argument("--full", action="store_true")
    p.add_argument("--no-open", action="store_true")
    p = sub.add_parser("review", help="review verified files")
    p.add_argument("--all", action="store_true", help="include files without indicators")
    sub.add_parser("restore", help="restore a quarantined file")
    p = sub.add_parser("train", help="generate synthetic data and retrain")
    p.add_argument("--epochs", type=int)
    return parser


def require_model():
    if not (config.OUTPUT_DIRECTORY / "model.pt").is_file():
        raise SystemExit("No trained model. Run: python main.py train")


def cli(argv):
    args = build_parser().parse_args(argv)
    if getattr(args, "top", None) is not None:
        if args.top < 0:
            raise SystemExit("--top must be 0 or more")
        config.TOP_K_FILES = args.top
    if args.command == "scan":
        require_model()
        from pathlib import Path
        from scanner import safe_path
        if not Path(args.folder).is_dir():
            raise SystemExit(f"Not a folder: {args.folder}")
        root = safe_path(args.folder, args.folder)  # Rejects links/junctions as roots.
        results = print_inspections(scan([root]), approved=not args.no_inspect)
        if args.html:
            open_report(full=False)
        if args.fail_on_pii and any(r.get("counts") for _, r in results):
            return 3
    elif args.command == "full":
        require_model()
        drives = fixed_drives()
        print("Fixed drives:", ", ".join(map(str, drives)))
        if not args.yes and input("Type SCAN to approve metadata scanning: ").strip() != "SCAN":
            print("Cancelled.")
            return 1
        candidates = scan(drives, full=True)
        if config.FULL_SCAN_ALLOW_CONTENT_INSPECTION:
            approved = args.inspect or input("Type INSPECT to approve reading only the top supported candidates: ").strip() == "INSPECT"
            print_inspections(candidates, approved)
        if args.html:
            open_report(full=True)
    elif args.command == "results":
        database = database_for(args.full)
        if not database.is_file():
            raise SystemExit("No scan results yet.")
        rows, total, threshold = query_results(database, high=args.high, extension=args.ext, folder=args.folder,
                                               changed=args.changed, limit=args.limit)
        print(f"{total:,} matching files (showing {len(rows)}):")
        print_rows(rows, threshold, multi_root=len(read_meta(database).get("roots", [])) > 1)
    elif args.command == "report":
        if args.no_open:
            from report import write_html_report
            print(write_html_report(full=args.full))
        else:
            open_report(full=args.full)
    elif args.command == "review":
        from review import review
        review(include_clean=args.all)
    elif args.command == "restore":
        from review import restore_interactive
        restore_interactive()
    elif args.command == "train":
        from generate_data import generate
        from train import train
        generate()
        train(args.epochs)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            sys.exit(cli(sys.argv[1:]))
        except (OSError, ValueError) as error:
            ui.error(str(error))
            sys.exit(1)
    main()
