"""One saved CNN and one encoder serve both modes; stream full reports to disk."""
import csv
import sqlite3
import time
from pathlib import Path
import torch
import config
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
        torch.set_num_threads(2)

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
    start, count = time.monotonic(), 0
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
            writer.writerow([str(path), str(root), score, score >= config.PREDICTION_THRESHOLD])
            connection.execute("INSERT INTO results VALUES (?, ?, ?)", (str(path), str(root), score))
        connection.commit()
        paths.flush()
        if config.SHOW_SCAN_PROGRESS:
            print(f"Evaluated {count} files; elapsed {time.monotonic() - start:.1f}s")
    with (config.OUTPUT_DIRECTORY / f"{prefix}file_paths.txt").open("w", encoding="utf-8") as paths, (config.OUTPUT_DIRECTORY / f"{prefix}results.csv").open("w", encoding="utf-8", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["absolute_path", "scan_root", "cnn_score", "above_threshold"])
        batch = []
        try:
            for root in roots:
                for path, boundary in enumerate_files(root, excluded):
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
            print("Cancelled; completed batches remain saved.")
    # SQLite sorts on disk: preserve incremental writes, then stream the final ranked CSV.
    connection.execute("CREATE INDEX score_order ON results(score DESC, path)")
    ranked_report = config.OUTPUT_DIRECTORY / f"{prefix}results.csv"
    with ranked_report.open("w", encoding="utf-8", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["absolute_path", "scan_root", "cnn_score", "above_threshold", "rank", "inspection_status"])
        for rank, (path, root, score) in enumerate(connection.execute("SELECT path, root, score FROM results ORDER BY score DESC, path"), 1):
            status = "Not inspected" if supported(path) else "Inspection unsupported"
            writer.writerow([path, root, score, score >= config.PREDICTION_THRESHOLD, rank, status])
    top = list(connection.execute("SELECT path, root, score FROM results ORDER BY score DESC, path LIMIT ?", (config.TOP_K_FILES,)))
    connection.commit()
    connection.close()
    if not full and not cancelled:
        save_tree(roots[0], config.OUTPUT_DIRECTORY / "tree.txt")
    print(f"Scan {'cancelled' if cancelled else 'complete'}: {count} files in {time.monotonic() - start:.1f}s")
    print("PII RISK ASSESSMENT (scores are not calibrated probabilities)")
    for rank, (path, root, score) in enumerate(top, 1):
        print(f"Rank: {rank}\nFile: {path}\nRisk Score: {score:.4f}\n")
        if not supported(path):
            print("Status: Inspection unsupported\n")
    return RankedCandidates(database)


def print_inspections(candidates, approved):
    for candidate, result in inspect_candidates(candidates, approved):
        print(f"File: {candidate['path']}\nCNN Risk Score: {candidate['score']:.4f}")
        for category, found in result["indicators"].items():
            print(f"{category} Indicators: {'Yes' if found else 'No'}")
        print("Status:", result["status"])
        if result.get("truncated"):
            print("Inspection limited to configured byte cap.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan an explicitly authorized test directory")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print_inspections(scan([args.root]), approved=True)
