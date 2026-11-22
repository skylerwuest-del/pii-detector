"""Small standard-library terminal helpers: color, tables, bars and progress.

Output is plain ASCII so it renders on any Windows console code page. Color is
used only on an interactive terminal and is disabled by the NO_COLOR variable.
"""
import os
import shutil
import sys
import time

CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33",
         "blue": "34", "magenta": "35", "cyan": "36", "gray": "90"}


_ansi_state = []  # Console mode is set once, not on every colored string.


def _enable_windows_ansi():
    if os.name != "nt":
        return True
    if _ansi_state:
        return _ansi_state[0]
    _ansi_state.append(False)
    try:
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        _ansi_state[0] = bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))  # VT processing.
    except Exception:
        pass
    return _ansi_state[0]


def color_enabled():
    return (sys.stdout.isatty() and "NO_COLOR" not in os.environ
            and os.environ.get("TERM") != "dumb" and _enable_windows_ansi())


def style(text, *names):
    if not names or not color_enabled():
        return str(text)
    return "\033[" + ";".join(CODES[n] for n in names) + "m" + str(text) + "\033[0m"


def width():
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 110))


def rule(char="-"):
    print(style(char * width(), "gray"))


def banner(title, subtitle=""):
    line = "=" * width()
    print(style(line, "cyan"))
    print(style(" " + title, "bold", "cyan"))
    if subtitle:
        print(style(" " + subtitle, "gray"))
    print(style(line, "cyan"))


def heading(title):
    print("\n" + style(f"== {title} ", "bold") + style("=" * max(0, width() - len(title) - 4), "gray"))


def ok(message):
    print(style("[ OK ] ", "green", "bold") + message)


def warn(message):
    print(style("[WARN] ", "yellow", "bold") + message)


def error(message):
    print(style("[FAIL] ", "red", "bold") + message)


def info(message):
    print(style("[INFO] ", "cyan") + message)


def shorten(text, limit):
    """Keep the END of a path: the filename matters more than the prefix."""
    text = str(text)
    return text if len(text) <= limit else "..." + text[-(limit - 3):]


def human_size(count):
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def human_time(seconds):
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def bar(fraction, size=20):
    fraction = min(1.0, max(0.0, float(fraction)))
    filled = int(round(fraction * size))
    return "[" + "#" * filled + "." * (size - filled) + "]"


def risk_label(score, threshold):
    """Color-coded band relative to the active decision threshold."""
    if score >= max(threshold, 0.8):
        return style("HIGH  ", "red", "bold")
    if score >= threshold:
        return style("MEDIUM", "yellow", "bold")
    return style("LOW   ", "green")


def table(headers, rows, aligns=None):
    """Print a simple aligned table. Cells may contain color codes (ignored for width)."""
    import re
    strip = re.compile(r"\033\[[0-9;]*m")
    plain = lambda cell: strip.sub("", str(cell))
    aligns = aligns or ["<"] * len(headers)
    sizes = [max([len(plain(h))] + [len(plain(r[i])) for r in rows]) for i, h in enumerate(headers)]
    def pad(cell, i):
        gap = sizes[i] - len(plain(cell))
        return " " * gap + str(cell) if aligns[i] == ">" else str(cell) + " " * gap
    print(style("  ".join(pad(h, i) for i, h in enumerate(headers)), "bold"))
    print(style("  ".join("-" * s for s in sizes), "gray"))
    for row in rows:
        print("  ".join(pad(cell, i) for i, cell in enumerate(row)))


class Progress:
    """One self-updating line on a terminal; sparse plain lines when piped.

    total=None shows a count and rate only (a directory scan has no known size).
    """
    def __init__(self, label, total=None, enabled=True, interval=0.1):
        self.label, self.total, self.enabled = label, total, enabled
        self.interval = interval if sys.stdout.isatty() else 5.0
        self.start = time.monotonic()
        self.last = 0.0
        self.live = sys.stdout.isatty()
        self.drew = False

    def update(self, done, extra=""):
        now = time.monotonic()
        if not self.enabled or now - self.last < self.interval:
            return
        self.last = now
        elapsed = max(now - self.start, 1e-9)
        rate = done / elapsed
        if self.total:
            eta = (self.total - done) / rate if rate else 0
            text = f"{self.label} {bar(done / self.total)} {done}/{self.total} {rate:,.0f}/s eta {human_time(eta)}"
        else:
            text = f"{self.label} {done:,} items  {rate:,.0f}/s  {human_time(elapsed)}"
        if extra:
            text += "  " + extra
        self._draw(text)

    def _draw(self, text):
        if self.live:
            text = shorten(text, width() - 1)
            print("\r" + text.ljust(width() - 1), end="", flush=True)
            self.drew = True
        else:
            print(text, flush=True)

    def finish(self, message=""):
        if self.drew:
            print("\r" + " " * (width() - 1) + "\r", end="")
        if message:
            print(message)
        self.drew = False


def line_chart(series, height=8, columns=56):
    """ASCII line chart. series: {symbol: [values]}; overlapping points print '*'."""
    values = [v for line in series.values() for v in line]
    if not values:
        return
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    grid = [[" "] * columns for _ in range(height)]
    for symbol, line in series.items():
        for column in range(columns):
            index = round(column * (len(line) - 1) / max(1, columns - 1)) if len(line) > 1 else 0
            row = height - 1 - round((line[index] - low) / span * (height - 1))
            grid[row][column] = symbol if grid[row][column] in (" ", symbol) else "*"
    for row, cells in enumerate(grid):
        label = high - span * row / max(1, height - 1)
        print(f"{label:7.3f} |" + "".join(cells))
    print(" " * 8 + "+" + "-" * columns)
