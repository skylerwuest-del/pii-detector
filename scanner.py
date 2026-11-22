"""Safe metadata enumeration and the controlled-mode navigator. Never executes shell input."""
import ctypes
import os
from pathlib import Path
import stat
import subprocess
import config


def is_link(path):
    # Windows junctions and other reparse points are skipped as well as symlinks.
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def within(path, root):
    return path.resolve().is_relative_to(root.resolve())


def safe_path(path, root):
    """Validate containment AND each existing component before opening a file."""
    path, root = Path(path).absolute(), Path(root).absolute()
    if is_link(root):
        raise ValueError("Linked roots are not allowed")
    root = root.resolve(strict=True)
    if not within(path, root):
        raise ValueError("Path escapes selected root")
    # Check lexical components too: resolving alone would hide links inside the root.
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if is_link(current):
            raise ValueError("Links and junctions are skipped")
    return path.resolve(strict=True)


class Navigator:
    def __init__(self, start=None):
        self.current = Path(start or Path.home()).resolve(strict=True)
        self.root = None
        self.previous = None

    def cd(self, destination):
        if str(destination) == "-":  # Like a shell: return to the previous directory.
            if self.previous is None:
                raise ValueError("No previous directory")
            destination = self.previous
        target = Path(destination).expanduser()
        if not target.is_absolute():
            target = self.current / target
        if self.root:
            target = safe_path(target, self.root)
        else:
            target = target.resolve(strict=True)
        if not target.is_dir():
            raise ValueError("Not a directory")
        self.previous, self.current = self.current, target

    def select(self):
        if self.root is not None:
            raise ValueError("Return to the main menu to select a different root")
        if is_link(self.current):
            raise ValueError("Cannot select a link")
        self.root = self.current.resolve(strict=True)

    def listing(self):
        """Directories first, then files: (name, is_dir, size, is_link). Metadata only."""
        entries = []
        with os.scandir(self.current) as scanned:
            for entry in scanned:
                try:
                    info = entry.stat(follow_symlinks=False)
                    linked = is_link(Path(entry.path))
                    folder = entry.is_dir(follow_symlinks=False)
                    entries.append((entry.name, folder, 0 if folder else info.st_size, linked))
                except OSError:
                    entries.append((entry.name, False, 0, False))
        return sorted(entries, key=lambda e: (not e[1], e[0].lower()))

    def tree_lines(self, depth=2, limit=200):
        """Bounded text tree of the current directory; links are shown but not entered."""
        lines = []
        def walk(folder, prefix, remaining):
            try:
                with os.scandir(folder) as scanned:
                    entries = sorted(scanned, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
            except OSError:
                lines.append(prefix + "[unreadable]")
                return
            for index, entry in enumerate(entries):
                if len(lines) >= limit:
                    return
                last = index == len(entries) - 1
                linked = is_link(Path(entry.path))
                folder_entry = entry.is_dir(follow_symlinks=False)
                lines.append(prefix + ("`-- " if last else "|-- ") + entry.name
                             + ("/" if folder_entry else "") + (" [link, not followed]" if linked else ""))
                if folder_entry and not linked and remaining > 1:
                    walk(entry.path, prefix + ("    " if last else "|   "), remaining - 1)
        walk(self.current, "", max(1, depth))
        if len(lines) >= limit:
            lines.append(f"... (stopped after {limit} entries)")
        return lines


def common_places():
    """Existing well-known user folders: handy starting points for a scan."""
    home = Path.home()
    names = ("Documents", "Downloads", "Desktop", "Pictures", "OneDrive")
    return [home / name for name in names if (home / name).is_dir()]


def enumerate_files(root, excluded=(), stats=None):
    """Yield (path, root) for every regular file; see enumerate_entries."""
    for path, boundary, _, _ in enumerate_entries(root, excluded, stats):
        yield path, boundary


def enumerate_entries(root, excluded=(), stats=None):
    """Yield (path, root, size, mtime) metadata only, without reading contents or following links.

    `stats` (optional dict) counts what was skipped so the UI can explain coverage.
    Each folder is validated once with safe_path; entries are checked as they are read.
    """
    def count(key):
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1
    root = Path(root).absolute()
    if config.FOLLOW_SYMLINKS:
        raise ValueError("FOLLOW_SYMLINKS must remain False")
    try:
        if is_link(root):
            raise ValueError("Root cannot be a link or junction")
        root = root.resolve(strict=True)
    except OSError:
        count("unreadable")
        return  # An unavailable drive/root does not crash a broad scan.
    # String comparison is far cheaper than resolving every entry against every exclusion.
    exclusions = [os.path.normcase(str(Path(p).resolve())) for p in excluded]
    def is_excluded(path):
        key = os.path.normcase(path)
        return any(key == e or key.startswith(e + os.sep) for e in exclusions)
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            safe_path(folder, root)
            with os.scandir(folder) as entries:
                for entry in entries:
                    try:
                        path = Path(entry.path)
                        info = entry.stat(follow_symlinks=False)
                        if is_link(path):
                            count("links")
                            continue
                        if not config.INCLUDE_HIDDEN_FILES and (entry.name.startswith(".") or getattr(info, "st_file_attributes", 0) & 2):
                            count("hidden")
                            continue
                        if exclusions and is_excluded(entry.path):
                            count("excluded")
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(path)
                        elif stat.S_ISREG(info.st_mode):
                            yield path, root, info.st_size, info.st_mtime
                    except (OSError, ValueError):
                        count("unreadable")
                        continue
        except (OSError, ValueError):
            count("unreadable")
            continue


def fixed_drives():
    if os.name != "nt":
        raise OSError("Automatic fixed-drive discovery is Windows-only")
    mask = ctypes.windll.kernel32.GetLogicalDrives()
    return [Path(f"{chr(65 + i)}:/") for i in range(26)
            if mask & (1 << i) and ctypes.windll.kernel32.GetDriveTypeW(f"{chr(65 + i)}:\\") == 3]


def broad_exclusions(roots):
    # Avoid system trees by default, without attempting to bypass their permissions.
    defaults = [r / name for r in roots for name in
                ("Windows", "Program Files", "Program Files (x86)", "ProgramData",
                 "$Recycle.Bin", "System Volume Information")]
    return defaults + [Path(p) for p in config.EXCLUDED_DIRECTORIES]


def save_tree(root, output):
    """Run Windows tree /F /A only after ensuring there are no reparse points.

    tree has no no-follow flag, so linked/unreadable trees use safe Python output.
    This extra walk checks directories as well as file links before invoking tree.
    """
    safe_for_tree = True
    def failed(_):
        nonlocal safe_for_tree
        safe_for_tree = False
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=failed):
        for name in dirs + files:
            try:
                if is_link(Path(folder) / name):
                    safe_for_tree = False
            except OSError:
                safe_for_tree = False
        safe_dirs = []
        for name in dirs:
            try:
                if not is_link(Path(folder) / name):
                    safe_dirs.append(name)
            except OSError:
                safe_for_tree = False
        dirs[:] = safe_dirs
    if os.name == "nt" and safe_for_tree:
        try:
            with output.open("w", encoding="utf-8") as handle:
                subprocess.run(["tree.com", str(root), "/F", "/A"], stdout=handle,
                               stderr=subprocess.DEVNULL, check=True, shell=False)
            return
        except (OSError, subprocess.SubprocessError):
            pass
    with output.open("w", encoding="utf-8") as handle:
        handle.write(str(root) + "\n[Safe Python hierarchy; links omitted]\n")
        for path, _ in enumerate_files(root, [config.OUTPUT_DIRECTORY]):
            handle.write("  " * len(path.relative_to(root).parts) + str(path.relative_to(root)) + "\n")
