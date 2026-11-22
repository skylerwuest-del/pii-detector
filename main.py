"""Windows terminal menu. Broad scans always separate metadata and content approval."""
import datetime
import json
import config
import ui
from scanner import Navigator, fixed_drives, common_places
from inference import scan, print_inspections

COMMANDS = [
    ("ls", "List this folder (folders first, with sizes)"),
    ("cd <folder>", "Change folder; 'cd ..' up, 'cd ~' home, 'cd -' previous"),
    ("pwd", "Show the current folder"),
    ("tree [depth]", "Show a bounded tree of this folder (default depth 2)"),
    ("places", "Jump list: Documents, Downloads, Desktop ..."),
    ("select", "Lock THIS folder as the authorized scan root"),
    ("scan", "Rank every file under the locked root, then verify the top ones"),
    ("help", "Show this list"),
    ("back", "Return to the main menu"),
    ("exit", "Quit"),
]
HELP = " | ".join(name for name, _ in COMMANDS)
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


def navigate():
    nav = Navigator()
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
                for place in places or []:
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
                    ui.info(f"Scanning locked root {nav.root}  (Ctrl+C cancels)")
                    print_inspections(scan([nav.root]), approved=True)
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
        ["Test examples", m.get("test_examples", "?")],
        ["Epochs run / best epoch", f"{m.get('epochs_run', len(m['history']))} / {m['best_epoch']}"],
        ["Seed", m["seed"]],
        ["Training time", ui.human_time(m.get("seconds", 0))],
    ], aligns=["<", ">"])
    print("\nSynthetic-data results do not prove real-world accuracy. Retrain after changing config.py model settings.")


def train_model():
    from generate_data import generate
    from train import train
    checkpoint = config.OUTPUT_DIRECTORY / "model.pt"
    if checkpoint.is_file() and input("This replaces the saved checkpoint. Retrain? [y/N]: ").strip().lower() != "y":
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


def main():
    while True:
        ui.banner("CS471 - PII DETECTION AGENT", "Character-level CNN ranks file paths; deterministic rules verify the top files")
        print(model_status()[1], "\n")
        print("  [1] Controlled directory scan   - pick a folder you are authorized to audit")
        print("  [2] Full computer scan          - all fixed drives (separate approvals)")
        print("  [3] Train / retrain the model")
        print("  [4] Model details")
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
            elif selection in ("0", "q", "quit", "exit"):
                return
            elif selection not in ("1", "2"):
                ui.warn(f"'{selection}' is not a menu option.")
        except (OSError, ValueError) as error:
            ui.error(f"Unable to continue: {error}")
        except (EOFError, KeyboardInterrupt):
            return
        print()


if __name__ == "__main__":
    main()
