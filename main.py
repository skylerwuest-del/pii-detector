"""Windows terminal menu. Broad scans always separate metadata and content approval."""
import config
from scanner import Navigator, fixed_drives
from inference import scan, print_inspections

HELP = "ls | cd <directory> | cd .. | cd ~ | pwd | select | scan | help | back | exit"


def navigate():
    nav = Navigator()
    print(HELP)
    print("Select only a synthetic test directory or a directory you are authorized to audit.")
    while True:
        try:
            command = input(f"{nav.current}> ").strip()
            action, _, argument = command.partition(" ")
            if action == "ls":
                for path in sorted(nav.current.iterdir()):
                    print(path.name)
            elif action == "cd":
                nav.cd(argument.strip().strip('"'))
            elif action == "pwd":
                print(nav.current)
            elif action == "select":
                nav.select()
                print("Locked root:", nav.root)
            elif action == "scan":
                if nav.root is None:
                    print("Use select to lock an authorized scan root first.")
                else:
                    print_inspections(scan([nav.root]), approved=True)
            elif action == "help":
                print(HELP)
            elif action == "back":
                return
            elif action == "exit":
                raise SystemExit
            else:
                print(HELP)
        except (OSError, ValueError) as error:
            print("Operation failed:", error)
        except KeyboardInterrupt:
            print("\nUse back to return or exit to quit.")


def full_mode():
    drives = fixed_drives()
    print("Detected local fixed drives:", ", ".join(map(str, drives)))
    print("Only proceed on a computer you own or are explicitly authorized to audit. Ctrl+C cancels.")
    if not drives:
        return
    if config.FULL_SCAN_REQUIRE_CONFIRMATION and input("Type SCAN to approve metadata scanning: ").strip() != "SCAN":
        return
    candidates = scan(drives, full=True)
    if config.FULL_SCAN_ALLOW_CONTENT_INSPECTION:
        approved = input("Type INSPECT to approve reading only the top supported candidates: ").strip() == "INSPECT"
        print_inspections(candidates, approved)


def main():
    while True:
        print("========================================\n CS471 - PII DETECTION AGENT\n========================================\n\nSelect Scanning Mode:\n\n[1] Controlled Directory Scan\n[2] Full Computer Scan\n[3] Exit\n")
        try:
            selection = input("Enter Selection: ").strip() or str(config.DEFAULT_SCAN_MODE)
            if selection == "1":
                navigate()
            elif selection == "2":
                full_mode()
            elif selection == "3":
                return
        except (OSError, ValueError) as error:
            print("Unable to continue:", error, "\nGenerate data and train before scanning.")
        except (EOFError, KeyboardInterrupt):
            return


if __name__ == "__main__":
    main()
