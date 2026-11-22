"""CPU training, validation checkpoint selection, and held-out evaluation.

Improvements over plain Adam training: class-balanced loss, AdamW weight decay,
gradient clipping, learning-rate reduction on plateaus, early stopping, per-epoch
path augmentation, a validation-tuned decision threshold, richer metrics, and two
learning checks (a shuffled-label control and a repeated-seed study).
"""
import argparse
import copy
import json
import random
import re
import statistics
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import config
import ui
from model import PathCNN, model_settings
from preprocessing import PathEncoder, metadata_features
from generate_data import build_families

# Neutral parent folders and file-name habits seen on real computers. They carry no label,
# so adding them teaches the CNN to ignore context and focus on the informative fragments.
NEUTRAL_DIRECTORIES = ("users/alex", "documents", "downloads", "desktop", "backup", "old files",
                       "2023", "2024", "projects", "shared", "onedrive", "new folder", "temp", "d")
COPY_SUFFIXES = (" (1)", " - copy", "_copy", "_final", "_v2", "_old", "-backup")


def augment_path(path, rng):
    """Training-only variation of a relative path. The label is never changed."""
    chance = config.AUGMENT_PROBABILITY
    directory, slash, name = path.rpartition("/")
    stem, dot, extension = name.rpartition(".")
    if not dot:
        stem, extension = name, ""
    if rng.random() < chance:  # Real files use different separators: employee_list / employee-list.
        separator = rng.choice(["_", "-", " "])
        stem = stem.replace("_", separator)
    if rng.random() < chance / 2:  # Duplicated or versioned files are common.
        stem += rng.choice(COPY_SUFFIXES)
    name = stem + (dot + extension if dot else "")
    path = directory + slash + name
    if rng.random() < chance:  # Real scans have deeper, unrelated parent folders.
        prefix = "/".join(rng.choice(NEUTRAL_DIRECTORIES) for _ in range(rng.randint(1, 3)))
        path = prefix + "/" + path
    if rng.random() < chance / 2 and path:  # One typo, as in hand-named files.
        index = rng.randrange(len(path))
        path = path[:index] + rng.choice(config.CHARACTERS) + path[index + 1:]
    return path


def metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / max(1e-12, precision + recall),
                accuracy=(tp + tn) / max(1, len(labels)), specificity=tn / max(1, tn + fp),
                confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn))


def fbeta(precision, recall, beta):
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0


def roc_auc(labels, scores):
    """Probability a random positive outranks a random negative (ties count half)."""
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def tune_threshold(labels, scores, beta):
    """Pick the cutoff with the best F-beta on VALIDATION data.

    Many cutoffs tie on a small set, so take the median of the tied range rather than
    an arbitrary edge; this is steadier on unseen paths.
    """
    candidates = [i / 100 for i in range(5, 96)]
    results = []
    for cutoff in candidates:
        m = metrics(labels, [int(s >= cutoff) for s in scores])
        results.append(fbeta(m["precision"], m["recall"], beta))
    best = max(results)
    tied = [c for c, r in zip(candidates, results) if r >= best - 1e-9]
    return float(statistics.median(tied)), best


def predict_logits(model, x, extra, batch_size):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_extra = extra[start:start + batch_size] if extra is not None else None
            logits.append(model(x[start:start + batch_size], batch_extra))
    return torch.cat(logits) if logits else torch.zeros(0)


def predict_scores(model, x, batch_size, extra=None, temperature=1.0):
    return torch.sigmoid(predict_logits(model, x, extra, batch_size) / temperature).tolist()


def mean_loss(model, x, y, loss_function, batch_size, extra=None):
    return loss_function(predict_logits(model, x, extra, batch_size), y).item() if len(x) else 0.0


def fit_temperature(logits, labels):
    """One scalar T so sigmoid(logit / T) matches validation frequencies (temperature scaling).

    T > 1 softens over-confident scores. Ranking, and therefore AUC, is unchanged.
    Only T >= 1 is allowed: a nearly separable validation set always "asks" for sharper
    scores, which then over-state confidence on harder unseen paths.
    """
    y = torch.tensor(labels, dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss()
    candidates = [1.1 ** i for i in range(50)]  # 1 .. ~107, geometric grid.
    return min(candidates, key=lambda t: loss(logits / t, y).item())


def calibration_error(labels, scores, bins=10):
    """Expected calibration error and per-bin (count, mean score, positive rate)."""
    table = []
    total_error = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [(y, s) for y, s in zip(labels, scores) if low <= s < high or (b == bins - 1 and s == 1.0)]
        if members:
            mean_score = sum(s for _, s in members) / len(members)
            positive_rate = sum(y for y, _ in members) / len(members)
            total_error += len(members) / max(1, len(labels)) * abs(mean_score - positive_rate)
            table.append(dict(bin=f"{low:.1f}-{high:.1f}", count=len(members), mean_score=mean_score, positive_rate=positive_rate))
    return total_error, table


TOKEN_SPLIT = re.compile(r"[/_\-. ()\d]+")


def error_analysis(records, labels, predictions, family_names):
    """Which families and words the test mistakes come from (helps decide what data to add)."""
    by_family, tokens = {}, {"false_alarm": {}, "missed": {}}
    for record, y, p in zip(records, labels, predictions):
        if y == p:
            continue
        kind = "false_alarm" if p else "missed"
        name = family_names.get(record.get("family"), str(record.get("family")))
        entry = by_family.setdefault(name, dict(family=name, label=y, false_alarms=0, missed=0, examples=0))
        entry["false_alarms" if p else "missed"] += 1
        for token in {t for t in TOKEN_SPLIT.split(record["path"].lower()) if len(t) > 2 and t not in ("txt", "csv", "tsv", "log", "json", "md")}:
            tokens[kind][token] = tokens[kind].get(token, 0) + 1
    totals = {}
    for record in records:
        name = family_names.get(record.get("family"), str(record.get("family")))
        totals[name] = totals.get(name, 0) + 1
    for name, entry in by_family.items():
        entry["examples"] = totals.get(name, 0)
    families = sorted(by_family.values(), key=lambda e: -(e["false_alarms"] + e["missed"]))
    top = lambda d: sorted(d.items(), key=lambda kv: -kv[1])[:10]
    return dict(families=families, false_alarm_words=top(tokens["false_alarm"]), missed_words=top(tokens["missed"]))


def load_dataset():
    path = config.DATA_DIRECTORY / "dataset.json"
    if not path.is_file():
        raise OSError(f"Training data not found at {path}. Run generate_data.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def feedback_records():
    """Your review decisions as extra training examples (never validation/test, which stay synthetic)."""
    if not config.USE_FEEDBACK:
        return []
    from review import load_feedback
    entries = [dict(path=e["path"], label=int(e["label"]), size=e.get("size") or 0, family="feedback")
               for e in load_feedback() if e.get("path")]
    return entries * max(1, config.FEEDBACK_REPEAT)


def run_training(seed, epochs, checkpoint=None, shuffle_labels=False, quiet=False, data=None):
    """Train one CNN. Returns a results dict; saves weights only when `checkpoint` is given."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.NUM_THREADS)
    rng = random.Random(seed)
    data = data or load_dataset()
    encoder = PathEncoder(strip_accents=config.STRIP_ACCENTS)
    settings = model_settings()
    use_metadata = settings["num_extra_features"] > 0
    encode = lambda paths: torch.tensor([encoder.encode(p) for p in paths], dtype=torch.long)
    def extras(records, paths=None):
        if not use_metadata:
            return None
        paths = paths or [r["path"] for r in records]
        return torch.tensor([metadata_features(p, r.get("size", 0)) for p, r in zip(paths, records)], dtype=torch.float32)
    feedback = feedback_records()
    train_records = list(data["train"]) + feedback
    labels = {split: [r["label"] for r in data[split]] for split in ("validation", "test")}
    train_labels = [r["label"] for r in train_records]
    if shuffle_labels:  # Control run: a model that learns from labels must fail on shuffled ones.
        rng.shuffle(train_labels)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    x = {s: encode([r["path"] for r in data[s]]) for s in ("validation", "test")}
    extra = {s: extras(data[s]) for s in ("validation", "test")}
    y = {s: torch.tensor(labels[s], dtype=torch.float32) for s in ("validation", "test")}
    model = PathCNN(**settings)
    initial = model.conv.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.LR_FACTOR, patience=config.LR_PATIENCE)
    positives = sum(train_labels)
    # Weighting the rarer class keeps a skewed corpus from teaching "always predict the majority".
    pos_weight = torch.tensor([(len(train_labels) - positives) / max(1, positives)]) if config.BALANCE_CLASSES else None
    # Binary cross-entropy penalizes wrong binary predictions; logits version is numerically stable.
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    best_key, best_auc = None, float("nan")  # Keys compare as "larger is better".
    by_auc = config.CHECKPOINT_BY == "auc"
    history, started, stopped_early = [], time.monotonic(), False
    if not quiet:
        ui.info(f"Training on {len(train_labels)} paths ({positives} positive"
                + (f", incl. {len(feedback)} from your feedback" if feedback else "")
                + f"), validating on {len(labels['validation'])}, testing on {len(labels['test'])}; seed {seed}"
                + (" [SHUFFLED-LABEL CONTROL]" if shuffle_labels else "")
                + (" [+ metadata features]" if use_metadata else ""))
        print(ui.style(f"{'epoch':>5}  {'':22}  {'train':>7}  {'valid':>7}  {'val AUC':>7}  {'val F1':>6}  {'lr':>8}", "bold"))
    try:
        for epoch in range(1, epochs + 1):
            paths = [r["path"] for r in train_records]
            if config.AUGMENT_TRAINING_PATHS:
                paths = [augment_path(p, rng) for p in paths]
            train_extra = extras(train_records, paths)
            tensors = (encode(paths), train_extra, y_train) if use_metadata else (encode(paths), y_train)
            loader = DataLoader(TensorDataset(*tensors), batch_size=config.BATCH_SIZE, shuffle=True)
            model.train()
            total = 0.0
            for batch in loader:
                batch_x, batch_y = batch[0], batch[-1]
                loss = loss_function(model(batch_x, batch[1] if use_metadata else None), batch_y)
                optimizer.zero_grad()
                loss.backward()  # Chain rule computes derivatives through CNN operations.
                if config.GRADIENT_CLIP:
                    nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()  # Update embedding, convolution and output weights.
                total += loss.item() * len(batch_y)
            validation_loss = mean_loss(model, x["validation"], y["validation"], loss_function, config.BATCH_SIZE, extra["validation"])
            epoch_scores = predict_scores(model, x["validation"], config.BATCH_SIZE, extra["validation"])
            validation_f1 = metrics(labels["validation"], [int(s >= 0.5) for s in epoch_scores])["f1"]
            validation_auc = roc_auc(labels["validation"], epoch_scores)
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step(validation_loss)
            record = dict(train=total / len(train_labels), validation=validation_loss,
                          validation_f1=validation_f1, validation_auc=validation_auc, learning_rate=learning_rate)
            history.append(record)
            # Ranking quality (AUC) is what the scanner relies on and is less noisy than log-loss,
            # which punishes confident mistakes on unseen families even when the ordering is good.
            key = (round(validation_auc, 3), -validation_loss) if by_auc else (-validation_loss,)
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_loss, best_auc, best_epoch, stale = key, validation_loss, validation_auc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if not quiet:
                print(f"{epoch:>5}  {ui.bar(epoch / epochs, 22)}  {record['train']:7.4f}  {validation_loss:7.4f}  "
                      f"{validation_auc:7.3f}  {validation_f1:6.3f}  {learning_rate:8.1e}" + (ui.style("  best", "green") if improved else ""))
            if config.EARLY_STOPPING_PATIENCE and stale >= config.EARLY_STOPPING_PATIENCE:
                stopped_early = True
                if not quiet:
                    ui.warn(f"Early stopping: no validation improvement for {stale} epochs.")
                break
    except KeyboardInterrupt:
        stopped_early = True
        if not quiet:
            ui.warn("Interrupted; keeping the best checkpoint found so far.")
    if best_state is None:
        raise RuntimeError("Training stopped before any epoch finished")
    model.load_state_dict(best_state)
    weight_change = (initial - model.conv.weight.detach()).abs().max().item()
    # Calibration and threshold use validation data only; the test set is touched exactly once.
    validation_logits = predict_logits(model, x["validation"], extra["validation"], config.BATCH_SIZE)
    temperature = fit_temperature(validation_logits, labels["validation"]) if config.CALIBRATE_SCORES else 1.0
    validation_scores = torch.sigmoid(validation_logits / temperature).tolist()
    tuned, tuned_score = tune_threshold(labels["validation"], validation_scores, config.THRESHOLD_F_BETA)
    threshold = tuned if config.AUTO_TUNE_THRESHOLD else config.PREDICTION_THRESHOLD
    test_logits = predict_logits(model, x["test"], extra["test"], config.BATCH_SIZE)
    test_scores = torch.sigmoid(test_logits / temperature).tolist()
    test_predictions = [int(s >= threshold) for s in test_scores]
    results = metrics(labels["test"], test_predictions)
    at_default = metrics(labels["test"], [int(s >= 0.5) for s in test_scores])
    test_positive_rate = sum(labels["test"]) / max(1, len(labels["test"]))
    uncalibrated_ece, _ = calibration_error(labels["test"], torch.sigmoid(test_logits).tolist())
    ece, reliability = calibration_error(labels["test"], test_scores)
    family_names = {i: template for i, (template, _) in enumerate(build_families())}
    results.update(auc=roc_auc(labels["test"], test_scores), threshold=threshold, temperature=temperature,
                   calibration_error=ece, uncalibrated_calibration_error=uncalibrated_ece, reliability=reliability,
                   errors=error_analysis(data["test"], labels["test"], test_predictions, family_names),
                   f1_at_0_5=at_default["f1"], tuned_validation_fbeta=tuned_score,
                   test_loss=mean_loss(model, x["test"], y["test"], loss_function, config.BATCH_SIZE, extra["test"]),
                   baseline_all_positive_f1=2 * test_positive_rate / (1 + test_positive_rate) if test_positive_rate else 0.0,
                   max_conv_weight_change=weight_change, history=history, test_examples=len(labels["test"]),
                   seed=seed, best_epoch=best_epoch, best_validation_loss=best_loss, best_validation_auc=best_auc,
                   epochs_run=len(history), stopped_early=stopped_early, shuffled_labels=shuffle_labels,
                   feedback_examples=len(feedback) // max(1, config.FEEDBACK_REPEAT), metadata_features=use_metadata,
                   seconds=time.monotonic() - started)
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), model_settings=settings, encoder=encoder.settings(),
                        threshold=float(threshold), temperature=float(temperature)), checkpoint)
    return results


def report(results):
    ui.heading("Training summary")
    series = {"t": [h["train"] for h in results["history"]], "v": [h["validation"] for h in results["history"]]}
    print("Loss per epoch   t = training   v = validation   * = both")
    ui.line_chart(series)
    c = results["confusion"]
    print(f"{'':8} epoch 1{' ' * 40}epoch {results['epochs_run']}\n")
    ui.table(["Held-out test", "Value"], [
        ["Precision", f"{results['precision']:.4f}"], ["Recall", f"{results['recall']:.4f}"],
        ["F1", f"{results['f1']:.4f}"], ["ROC AUC", f"{results['auc']:.4f}"],
        ["Accuracy", f"{results['accuracy']:.4f}"],
        ["All-positive baseline F1", f"{results['baseline_all_positive_f1']:.4f}"],
        ["Decision threshold", f"{results['threshold']:.2f}" + (" (tuned on validation)" if config.AUTO_TUNE_THRESHOLD else " (config)")],
    ], aligns=["<", ">"])
    print(f"\nConfusion (test, n={results['test_examples']}):  caught PII {c['tp']}  missed PII {c['fn']}  "
          f"false alarms {c['fp']}  correct skips {c['tn']}")
    print(f"Selected epoch {results['best_epoch']} of {results['epochs_run']} run (by validation {config.CHECKPOINT_BY}), validation loss {results['best_validation_loss']:.4f}; "
          f"maximum convolution weight change {results['max_conv_weight_change']:.6f}; {ui.human_time(results['seconds'])}")
    if results.get("feedback_examples"):
        print(f"Included {results['feedback_examples']} feedback label(s) from your reviews in training.")
    last = results["history"][-1]
    if last["validation"] > min(h["validation"] for h in results["history"]) * 1.1:
        ui.warn("Validation loss worsened after the best epoch: evidence of overfitting. The best earlier checkpoint was retained.")
    if "reliability" in results:
        ui.heading("Calibration (test)")
        print(f"Temperature {results['temperature']:.2f} fitted on validation. Expected calibration error "
              f"{results['uncalibrated_calibration_error']:.3f} before -> {results['calibration_error']:.3f} after "
              "(0 = a score of 0.8 means PII 80% of the time).")
        ui.table(["Score bin", "Paths", "Mean score", "Actually PII", ""],
                 [[r["bin"], r["count"], f"{r['mean_score']:.2f}", f"{r['positive_rate']:.2f}", ui.bar(r["positive_rate"], 10)]
                  for r in results["reliability"]], aligns=["<", ">", ">", ">", "<"])
    errors = results.get("errors")
    if errors and errors["families"]:
        ui.heading("Where the test mistakes come from")
        ui.table(["Path family", "True label", "False alarms", "Missed", "Family size"],
                 [[ui.shorten(f["family"], 44), "PII" if f["label"] else "benign", f["false_alarms"], f["missed"], f["examples"]]
                  for f in errors["families"][:8]], aligns=["<", "<", ">", ">", ">"])
        if errors["false_alarm_words"]:
            print("Words in false alarms: " + ", ".join(f"{w} ({n})" for w, n in errors["false_alarm_words"][:8]))
        if errors["missed_words"]:
            print("Words in missed PII:   " + ", ".join(f"{w} ({n})" for w, n in errors["missed_words"][:8]))
        print("Add training families that separate these words by content to reduce such mistakes.")


def train(epochs=None, seed=None):
    config.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ui.heading("Training the path CNN")
    results = run_training(seed if seed is not None else config.RANDOM_SEED,
                           epochs or config.EPOCHS, config.OUTPUT_DIRECTORY / "model.pt")
    (config.OUTPUT_DIRECTORY / "training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report(results)
    ui.ok(f"Saved {config.OUTPUT_DIRECTORY / 'model.pt'}")
    return results


def control(epochs=None):
    """Shuffled-label control: same CNN, scrambled answers. It should score near the baseline."""
    ui.heading("Shuffled-label control")
    data = load_dataset()
    real = run_training(config.RANDOM_SEED, epochs or config.EPOCHS, quiet=True, data=data)
    shuffled = run_training(config.RANDOM_SEED, epochs or config.EPOCHS, shuffle_labels=True, quiet=True, data=data)
    ui.table(["Run", "Test F1", "AUC"], [["real labels", f"{real['f1']:.4f}", f"{real['auc']:.4f}"],
                                          ["shuffled labels", f"{shuffled['f1']:.4f}", f"{shuffled['auc']:.4f}"]], aligns=["<", ">", ">"])
    print("AUC near 0.50 for the shuffled run shows the real result is learned from the labels.")
    return real, shuffled


def seed_study(count, epochs=None):
    """Repeat training with different seeds on the same data to measure variability."""
    ui.heading(f"Repeated-seed study ({count} runs)")
    data = load_dataset()
    rows, runs = [], []
    for index in range(count):
        seed = config.RANDOM_SEED + index
        r = run_training(seed, epochs or config.EPOCHS, quiet=True, data=data)
        runs.append(r)
        rows.append([seed, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}", f"{r['auc']:.3f}", r["best_epoch"]])
        print(f"  seed {seed}: F1 {r['f1']:.3f}  AUC {r['auc']:.3f}", flush=True)
    print()
    ui.table(["Seed", "Precision", "Recall", "F1", "AUC", "Best epoch"], rows, aligns=[">"] * 6)
    for key in ("f1", "auc"):
        values = [r[key] for r in runs]
        print(f"{key.upper():>4}: mean {statistics.mean(values):.4f}  std {statistics.pstdev(values):.4f}  "
              f"min {min(values):.4f}  max {max(values):.4f}")
    (config.OUTPUT_DIRECTORY / "seed_study.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "history"} for r in runs], indent=2), encoding="utf-8")
    return runs


def sweep(epochs=None, grid=None):
    """Try every combination in SWEEP_GRID; choose by VALIDATION ranking, never by test scores."""
    import itertools
    grid = grid or config.SWEEP_GRID
    names = list(grid)
    combos = list(itertools.product(*(grid[n] for n in names)))
    ui.heading(f"Hyperparameter sweep ({len(combos)} runs)")
    data = load_dataset()
    original = {n: getattr(config, n) for n in names}
    runs = []
    try:
        for index, values in enumerate(combos, 1):
            for name, value in zip(names, values):
                setattr(config, name, value)
            if (config.MAX_PATH_LENGTH + 2 * config.PADDING - config.KERNEL_SIZE) // config.STRIDE + 1 < 1:
                continue  # This combination leaves no convolution output positions.
            r = run_training(config.RANDOM_SEED, epochs or config.EPOCHS, quiet=True, data=data)
            runs.append(dict(settings=dict(zip(names, values)), validation_auc=r["best_validation_auc"],
                             validation_loss=r["best_validation_loss"], test_f1=r["f1"], test_auc=r["auc"]))
            print(f"  {index:>2}/{len(combos)}  " + "  ".join(f"{n}={v}" for n, v in zip(names, values))
                  + f"  ->  val AUC {r['best_validation_auc']:.3f}  val loss {r['best_validation_loss']:.4f}", flush=True)
    finally:
        for name, value in original.items():
            setattr(config, name, value)
    runs.sort(key=lambda r: (-round(r["validation_auc"], 3), r["validation_loss"]))
    print()
    ui.table(names + ["Val AUC", "Val loss"],
             [[r["settings"][n] for n in names] + [f"{r['validation_auc']:.3f}", f"{r['validation_loss']:.4f}"] for r in runs[:10]],
             aligns=[">"] * (len(names) + 2))
    best = runs[0]
    ui.ok("Best on validation: " + ", ".join(f"{n} = {v}" for n, v in best["settings"].items())
          + f"  (its test F1 {best['test_f1']:.3f}, AUC {best['test_auc']:.3f}, reported once for reference)")
    print("To use it, set these values in config.py and retrain. Picking by test scores would overstate accuracy.")
    (config.OUTPUT_DIRECTORY / "sweep.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the path CNN (defaults come from config.py)")
    parser.add_argument("--epochs", type=int, help="override EPOCHS")
    parser.add_argument("--seed", type=int, help="override RANDOM_SEED for this run")
    parser.add_argument("--seeds", type=int, metavar="N", help="repeat N seeds on the same data; saves nothing")
    parser.add_argument("--control", action="store_true", help="shuffled-label control; saves nothing")
    parser.add_argument("--sweep", action="store_true", help="try SWEEP_GRID settings, pick by validation; saves nothing")
    args = parser.parse_args()
    config.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        sweep(args.epochs)
    elif args.seeds:
        seed_study(args.seeds, args.epochs)
    elif args.control:
        control(args.epochs)
    else:
        train(args.epochs, args.seed)
