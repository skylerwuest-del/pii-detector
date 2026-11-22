"""One saved CNN and one encoder serve both modes; stream full reports to disk."""
import csv
import sqlite3
import time
from pathlib import Path
import torch
import config
import ui
from model import PathCNN
from preprocessing import PathEncoder
from scanner import enumerate_files, save_tree, broad_exclusions
from inspection import supported, inspect_candidates


class Predictor:
    def __init__(self, checkpoint=None):
        saved = torch.load(checkpoint or config.OUTPUT_DIRECTORY / "model.pt", map_location="cpu", weights_only=True)
        self.encoder = PathEncoder(**saved["encoder"])
        self.model = PathCNN(**saved["model_settings"])
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(config.NUM_THREADS)
        # A validation-tuned cutoff travels with the checkpoint; older checkpoints lack one.
        self.tuned_threshold = saved.get("threshold")

    @property
    def threshold(self):
        if config.AUTO_TUNE_THRESHOLD and self.tuned_threshold is not None:
            return self.tuned_threshold
        return config.PREDICTION_THRESHOLD

    def scores(self, relative_paths):
        ids = torch.tensor([self.encoder.encode(p) for p in relative_paths], dtype=torch.long)
        with torch.no_grad():
            return torch.sigmoid(self.model(ids)).tolist()


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


def scan(roots, full=False, predictor=None):
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
    threshold = predictor.threshold
    progress = ui.Progress("Scanning", enabled=config.SHOW_SCAN_PROGRESS)
    database = config.OUTPUT_DIRECTORY / f"{prefix}ranking.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE IF EXISTS results")
    connection.execute("CREATE TABLE results (path TEXT, root TEXT, score REAL)")
    limit = config.MAX_FILES_TO_SCAN
    cancelled = False
    def process(batch, paths, writer):
        nonlocal count
        scores = predictor.scores([p.relative_to(r).as_posix() for p, r in batch])
        for (path, root), score in zip(batch, scores):
            count += 1
            paths.write(str(path) + "\n")
            writer.writerow([str(path), str(root), score, score >= threshold])
            connection.execute("INSERT INTO results VALUES (?, ?, ?)", (str(path), str(root), score))
        connection.commit()
        paths.flush()
        progress.update(count)
    with (config.OUTPUT_DIRECTORY / f"{prefix}file_paths.txt").open("w", encoding="utf-8") as paths, (config.OUTPUT_DIRECTORY / f"{prefix}results.csv").open("w", encoding="utf-8", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["absolute_path", "scan_root", "cnn_score", "above_threshold"])
        batch = []
        try:
            for root in roots:
                for path, boundary in enumerate_files(root, excluded, stats):
                    if limit is not None and count + len(batch) >= limit:
                        break
                    batch.append((path, boundary))
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
            ui.warn("Cancelled; completed batches remain saved.")
    progress.finish()
    # SQLite sorts on disk: preserve incremental writes, then stream the final ranked CSV.
    connection.execute("CREATE INDEX score_order ON results(score DESC, path)")
    ranked_report = config.OUTPUT_DIRECTORY / f"{prefix}results.csv"
    with ranked_report.open("w", encoding="utf-8", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["absolute_path", "scan_root", "cnn_score", "above_threshold", "rank", "inspection_status"])
        for rank, (path, root, score) in enumerate(connection.execute("SELECT path, root, score FROM results ORDER BY score DESC, path"), 1):
            status = "Not inspected" if supported(path) else "Inspection unsupported"
            writer.writerow([path, root, score, score >= threshold, rank, status])
    top = list(connection.execute("SELECT path, root, score FROM results ORDER BY score DESC, path LIMIT ?", (config.TOP_K_FILES,)))
    flagged = connection.execute("SELECT COUNT(*) FROM results WHERE score >= ?", (threshold,)).fetchone()[0]
    connection.commit()
    connection.close()
    if not full and not cancelled:
        save_tree(roots[0], config.OUTPUT_DIRECTORY / "tree.txt")
    elapsed = time.monotonic() - start
    ui.heading("Scan " + ("cancelled" if cancelled else "complete"))
    skipped = ", ".join(f"{n} {k}" for k, n in sorted(stats.items())) or "nothing"
    print(f"Evaluated {count:,} files in {ui.human_time(elapsed)} ({count / max(elapsed, 1e-9):,.0f}/s). "
          f"Skipped: {skipped}.")
    print(f"Above the {threshold:.2f} decision threshold: {ui.style(flagged, 'bold')} of {count:,} paths "
          f"({100 * flagged / max(1, count):.1f}%).")
    ui.heading("Highest-risk paths  (CNN scores are not calibrated probabilities)")
    rows = []
    for rank, (path, root, score) in enumerate(top, 1):
        shown = ui.shorten(Path(path).relative_to(root).as_posix() if len(roots) == 1 else path, 52)
        rows.append([rank, ui.risk_label(score, threshold), f"{score:.4f}", ui.bar(score, 10),
                     shown, "yes" if supported(path) else ui.style("no", "gray")])
    ui.table(["#", "Risk", "Score", "", "Path (relative to root)" if len(roots) == 1 else "Path", "Inspectable"],
             rows, aligns=[">", "<", ">", "<", "<", "<"])
    print(f"\nReports: {config.OUTPUT_DIRECTORY / (prefix + 'results.csv')}  |  paths: {config.OUTPUT_DIRECTORY / (prefix + 'file_paths.txt')}")
    return RankedCandidates(database)


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
        rows.append([candidate["path"], f"{candidate['score']:.4f}", result["status"], level,
                     ";".join(f"{n}={c}" for n, c in found.items()), result.get("truncated", False)])
    print()
    if rows:
        report = config.OUTPUT_DIRECTORY / "inspection_report.csv"
        config.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["absolute_path", "cnn_score", "status", "severity", "indicator_counts", "truncated"])
            writer.writerows(rows)
        (ui.warn if confirmed else ui.ok)(f"{confirmed} of {len(rows)} inspected files contain PII indicators. "
                                           f"Review before deleting anything: {report}")
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
