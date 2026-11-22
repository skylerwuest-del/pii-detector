"""One saved CNN and one encoder serve both modes; stream full reports to disk.

Each scan writes a ranking database (results + meta). A separate score history
remembers every file's size, modification time and score, so a later scan reuses
scores for unchanged files: an interrupted scan restarts quickly and the
"changed since last scan" view needs no extra work.
"""
import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import torch
import config
import ui
from model import PathCNN
from preprocessing import PathEncoder, metadata_features
from scanner import enumerate_entries, save_tree, broad_exclusions
from inspection import supported, inspect_candidates


class Predictor:
    def __init__(self, checkpoint=None):
        path = Path(checkpoint or config.OUTPUT_DIRECTORY / "model.pt")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        # Identity of these exact weights: saved scores are reused only for the same model.
        self.model_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.encoder = PathEncoder(**saved["encoder"])
        self.model = PathCNN(**saved["model_settings"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(config.NUM_THREADS)
        # A validation-tuned cutoff and temperature travel with the checkpoint; older ones lack them.
        self.tuned_threshold = saved.get("threshold")
        self.temperature = float(saved.get("temperature", 1.0))
        self.uses_metadata = self.model.num_extra_features > 0

    @property
    def threshold(self):
        if config.AUTO_TUNE_THRESHOLD and self.tuned_threshold is not None:
            return self.tuned_threshold
        return config.PREDICTION_THRESHOLD

    def scores(self, relative_paths, sizes=None):
        relative_paths = list(relative_paths)
        if not relative_paths:
            return []
        ids = torch.tensor([self.encoder.encode(p) for p in relative_paths], dtype=torch.long)
        extra = None
        if self.uses_metadata:
            sizes = sizes or [0] * len(relative_paths)
            extra = torch.tensor([metadata_features(p, s) for p, s in zip(relative_paths, sizes)], dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.model(ids, extra) / self.temperature).tolist()


class ScoreHistory:
    """Path -> (size, mtime, score, model) across scans, stored on disk."""
    def __init__(self, database):
        self.connection = sqlite3.connect(database)
        self.connection.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, "
                                "mtime REAL, score REAL, model_id TEXT, last_seen REAL)")

    def lookup(self, path):
        return self.connection.execute("SELECT size, mtime, score, model_id FROM files WHERE path = ?", (path,)).fetchone()

    def store(self, rows):
        self.connection.executemany("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)", rows)
        self.connection.commit()

    def close(self):
        self.connection.close()


class RankedCandidates:
    """Read sorted candidates from disk lazily, so even a drive scan stays bounded."""
    def __init__(self, database):
        self.database = database

    def __iter__(self):
        connection = sqlite3.connect(self.database)
        try:
            for path, root, score in connection.execute("SELECT path, root, score FROM results ORDER BY score DESC, path"):
                if supported(path):
                    yield dict(path=path, root=root, score=score)
        finally:
            connection.close()


def database_for(full):
    return config.OUTPUT_DIRECTORY / f"{'full_scan_' if full else ''}ranking.sqlite3"


def read_meta(database):
    connection = sqlite3.connect(database)
    try:
        return {k: json.loads(v) for k, v in connection.execute("SELECT key, value FROM meta")}
    except sqlite3.OperationalError:
        return {}
    finally:
        connection.close()


def write_meta(connection, **values):
    connection.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in values.items()])
    connection.commit()


def interrupted_scan(full=True):
    """Meta of a previous scan that did not finish, or None."""
    database = database_for(full)
    meta = read_meta(database) if database.is_file() else {}
    return meta if meta and not meta.get("complete") else None


def scan(roots, full=False, predictor=None, quiet=False):
    predictor = predictor or Predictor()  # Same class and checkpoint for both modes.
    roots = [Path(r).absolute() for r in roots]
    if config.TOP_K_FILES < 0 or config.INFERENCE_BATCH_SIZE < 1:
        raise ValueError("TOP_K_FILES must be nonnegative and INFERENCE_BATCH_SIZE positive")
    if config.MAX_FILES_TO_SCAN is not None and config.MAX_FILES_TO_SCAN < 0:
        raise ValueError("MAX_FILES_TO_SCAN must be None or nonnegative")
    config.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prefix = "full_scan_" if full else ""
    excluded = broad_exclusions(roots) if full else list(config.EXCLUDED_DIRECTORIES)
    excluded += [config.OUTPUT_DIRECTORY, config.DATA_DIRECTORY, config.PROJECT_DIRECTORY / ".venv", config.PROJECT_DIRECTORY / ".git"]
    start, count, stats = time.monotonic(), 0, {}
    statuses = {"new": 0, "changed": 0, "unchanged": 0}
    reused = 0
    threshold = predictor.threshold
    progress = ui.Progress("Scanning", enabled=config.SHOW_SCAN_PROGRESS and not quiet)
    database = database_for(full)
    history = ScoreHistory(config.OUTPUT_DIRECTORY / "scan_history.sqlite3")
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE IF EXISTS results")
    connection.execute("DROP TABLE IF EXISTS meta")
    connection.execute("CREATE TABLE results (path TEXT, root TEXT, score REAL, size INTEGER, mtime REAL, status TEXT)")
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    write_meta(connection, roots=[str(r) for r in roots], full=full, threshold=threshold, complete=False,
               started=time.time(), model_id=predictor.model_id)
    limit = config.MAX_FILES_TO_SCAN
    cancelled = False

    def process(batch, paths, writer):
        nonlocal count, reused
        scores, to_score = [None] * len(batch), []
        for index, (path, root, size, mtime) in enumerate(batch):
            saved = history.lookup(str(path))
            same_file = saved is not None and saved[0] == size and saved[1] == mtime
            statuses["new" if saved is None else "unchanged" if same_file else "changed"] += 1
            batch[index] = (path, root, size, mtime, "new" if saved is None else "unchanged" if same_file else "changed")
            if same_file and saved[3] == predictor.model_id and config.REUSE_UNCHANGED_SCORES:
                scores[index] = saved[2]
                reused += 1
            else:
                to_score.append(index)
        fresh = predictor.scores([batch[i][0].relative_to(batch[i][1]).as_posix() for i in to_score],
                                 [batch[i][2] for i in to_score])
        for index, score in zip(to_score, fresh):
            scores[index] = score
        now = time.time()
        rows = []
        for (path, root, size, mtime, status), score in zip(batch, scores):
            count += 1
            paths.write(str(path) + "\n")
            writer.writerow([str(path), str(root), score, score >= threshold])
            rows.append((str(path), str(root), score, size, mtime, status))
        connection.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
        history.store([(p, s, m, score, predictor.model_id, now) for p, _, score, s, m, _ in rows])
        paths.flush()
        progress.update(count, f"reused {reused:,}" if reused else "")

    with (config.OUTPUT_DIRECTORY / f"{prefix}file_paths.txt").open("w", encoding="utf-8") as paths, (config.OUTPUT_DIRECTORY / f"{prefix}results.csv").open("w", encoding="utf-8", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["absolute_path", "scan_root", "cnn_score", "above_threshold"])
        batch = []
        try:
            for root in roots:
                for entry in enumerate_entries(root, excluded, stats):
                    if limit is not None and count + len(batch) >= limit:
                        break
                    batch.append(entry)
                    if len(batch) >= config.INFERENCE_BATCH_SIZE:
                        process(batch, paths, writer)
                        report.flush()
                        batch = []
                if limit is not None and count + len(batch) >= limit:
                    break
            if batch:
                process(batch, paths, writer)
        except KeyboardInterrupt:
            cancelled = True
            progress.finish()
            ui.warn("Cancelled; completed batches remain saved. Scanning again reuses their scores.")
    progress.finish()
    history.close()
    # SQLite sorts on disk: preserve incremental writes, then stream the final ranked CSV.
    connection.execute("CREATE INDEX score_order ON results(score DESC, path)")
    ranked_report = config.OUTPUT_DIRECTORY / f"{prefix}results.csv"
    with ranked_report.open("w", encoding="utf-8", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["absolute_path", "scan_root", "cnn_score", "above_threshold", "rank", "inspection_status", "change_status"])
        for rank, (path, root, score, status) in enumerate(connection.execute("SELECT path, root, score, status FROM results ORDER BY score DESC, path"), 1):
            inspection = "Not inspected" if supported(path) else "Inspection unsupported"
            writer.writerow([path, root, score, score >= threshold, rank, inspection, status])
    elapsed = time.monotonic() - start
    write_meta(connection, complete=not cancelled, finished=time.time(), elapsed=elapsed, files=count,
               skipped=stats, statuses=statuses, reused=reused)
    connection.close()
    if not full and not cancelled:
        save_tree(roots[0], config.OUTPUT_DIRECTORY / "tree.txt")
    if not quiet:
        print_summary(database)
    return RankedCandidates(database)


# ---------- reading a finished ranking ----------
def like_escape(text):
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def query_results(database, high=False, extension=None, folder=None, changed=False, offset=0, limit=None):
    """Filtered, ranked rows: (rows, total, threshold). Filters combine with AND."""
    meta = read_meta(database)
    threshold = float(meta.get("threshold", config.PREDICTION_THRESHOLD))
    where, args = [], []
    if high:
        where.append("score >= ?")
        args.append(threshold)
    if extension:
        extension = extension.lower() if extension.startswith(".") else "." + extension.lower()
        where.append("lower(path) LIKE ? ESCAPE '\\'")
        args.append("%" + like_escape(extension))
    if folder:
        where.append("lower(replace(path, '\\', '/')) LIKE ? ESCAPE '\\'")
        args.append("%/" + like_escape(folder.lower().replace("\\", "/").strip("/")) + "/%")
    if changed:
        where.append("status != 'unchanged'")
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    connection = sqlite3.connect(database)
    try:
        total = connection.execute("SELECT COUNT(*) FROM results" + clause, args).fetchone()[0]
        rows = list(connection.execute("SELECT path, root, score, status FROM results" + clause
                                       + " ORDER BY score DESC, path LIMIT ? OFFSET ?",
                                       args + [limit if limit is not None else -1, offset]))
    finally:
        connection.close()
    return rows, total, threshold


def folder_risk(database, depth=1, limit=10):
    """Per-folder aggregate (folders `depth` levels below each root), riskiest first."""
    meta = read_meta(database)
    threshold = float(meta.get("threshold", config.PREDICTION_THRESHOLD))
    totals = {}
    connection = sqlite3.connect(database)
    try:
        for path, root, score in connection.execute("SELECT path, root, score FROM results"):
            try:
                parts = Path(path).relative_to(root).parts[:-1][:max(1, depth)]
            except ValueError:
                parts = ()
            key = (root, "/".join(parts) or "(top level)")
            entry = totals.setdefault(key, [0, 0, 0.0, 0.0])
            entry[0] += 1
            entry[1] += score >= threshold
            entry[2] += score
            entry[3] = max(entry[3], score)
    finally:
        connection.close()
    rows = [dict(root=root, folder=folder, files=n, flagged=f, mean=total / n, max=top)
            for (root, folder), (n, f, total, top) in totals.items()]
    rows.sort(key=lambda r: (-r["flagged"], -r["max"], -r["mean"]))
    return rows[:limit] if limit else rows


def print_rows(rows, threshold, offset=0, multi_root=False):
    table = []
    for rank, (path, root, score, status) in enumerate(rows, offset + 1):
        shown = ui.shorten(path if multi_root else Path(path).relative_to(root).as_posix(), 50)
        mark = {"new": ui.style("new", "cyan"), "changed": ui.style("changed", "yellow")}.get(status, "")
        table.append([rank, ui.risk_label(score, threshold), f"{score:.4f}", ui.bar(score, 10), shown,
                      "yes" if supported(path) else ui.style("no", "gray"), mark])
    ui.table(["#", "Risk", "Score", "", "Path" if multi_root else "Path (relative to root)", "Inspectable", "Change"],
             table, aligns=[">", "<", ">", "<", "<", "<", "<"])


def print_folders(database, depth=1, limit=5):
    rows = folder_risk(database, depth, limit)
    multi_root = len(read_meta(database).get("roots", [])) > 1
    ui.table(["Folder", "Files", "Above threshold", "Mean", "Max"],
             [[ui.shorten((r["root"] + " : " if multi_root else "") + r["folder"], 50), r["files"],
               r["flagged"], f"{r['mean']:.3f}", f"{r['max']:.3f}"] for r in rows],
             aligns=["<", ">", ">", ">", ">"])


def print_summary(database):
    meta = read_meta(database)
    rows, total, threshold = query_results(database, limit=config.TOP_K_FILES)
    flagged = query_results(database, high=True, limit=0)[1]
    multi_root = len(meta.get("roots", [])) > 1
    elapsed = meta.get("elapsed", 0)
    statuses = meta.get("statuses", {})
    ui.heading("Scan " + ("complete" if meta.get("complete") else "cancelled"))
    skipped = ", ".join(f"{n} {k}" for k, n in sorted(meta.get("skipped", {}).items())) or "nothing"
    print(f"Evaluated {total:,} files in {ui.human_time(elapsed)} ({total / max(elapsed, 1e-9):,.0f}/s). Skipped: {skipped}.")
    if meta.get("reused"):
        print(f"Reused saved scores for {meta['reused']:,} unchanged files.")
    print(f"Since the last scan: {statuses.get('new', 0):,} new, {statuses.get('changed', 0):,} changed, "
          f"{statuses.get('unchanged', 0):,} unchanged.")
    print(f"Above the {threshold:.2f} decision threshold: {ui.style(flagged, 'bold')} of {total:,} paths "
          f"({100 * flagged / max(1, total):.1f}%).")
    ui.heading("Highest-risk paths")
    print_rows(rows, threshold, multi_root=multi_root)
    if total:
        ui.heading("Riskiest folders")
        print_folders(database)
    prefix = "full_scan_" if meta.get("full") else ""
    print(f"\nReports: {config.OUTPUT_DIRECTORY / (prefix + 'results.csv')}  |  paths: {config.OUTPUT_DIRECTORY / (prefix + 'file_paths.txt')}")


def print_inspections(candidates, approved):
    """Read the top candidates, print a compact verdict per file, and save a values-free report."""
    results = inspect_candidates(candidates, approved)
    if not approved:
        ui.info("Content inspection declined: no file contents were opened.")
        return results
    ui.heading("Content verification (deterministic rules; matched values are never shown)")
    confirmed, rows = 0, []
    for candidate, result in results:
        found = result.get("counts", {})
        level = result.get("severity", "None")
        if found:
            label, styles = level.upper() + " RISK", {"High": ("red", "bold"), "Medium": ("yellow", "bold")}.get(level, ("yellow",))
        elif result["status"] == "Metadata clues detected":
            label, styles = "METADATA CLUE", ("yellow",)
        elif result["status"] == "Inspection failed":
            label, styles = "COULD NOT READ", ("gray",)
        else:
            label, styles = "NO INDICATORS", ("green",)
        print(f"{ui.style(f'{label:<14}', *styles)} {ui.shorten(candidate['path'], ui.width() - 16)}")
        detail = ", ".join(f"{name} x{count}" for name, count in found.items()) if found else result["status"]
        print(f"      CNN score {candidate['score']:.4f}  |  {detail}"
              + ("  |  truncated at byte cap" if result.get("truncated") else ""))
        if found:
            confirmed += 1
        rows.append([candidate["path"], candidate["root"], f"{candidate['score']:.4f}", result["status"], level,
                     ";".join(f"{n}={c}" for n, c in found.items()), result.get("truncated", False)])
    print()
    if rows:
        report = config.OUTPUT_DIRECTORY / "inspection_report.csv"
        config.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["absolute_path", "scan_root", "cnn_score", "status", "severity", "indicator_counts", "truncated"])
            writer.writerows(rows)
        (ui.warn if confirmed else ui.ok)(f"{confirmed} of {len(rows)} inspected files contain PII indicators. "
                                           f"Use `review` to go through them: {report}")
        print("No indicators does not prove a file is free of PII; rules can miss formats and encodings.")
    else:
        ui.info("No readable candidate files to inspect.")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan an explicitly authorized test directory")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print_inspections(scan([args.root]), approved=True)
