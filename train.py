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
import statistics
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import config
import ui
from model import PathCNN, model_settings
from preprocessing import PathEncoder

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


def predict_scores(model, x, batch_size):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            scores.extend(torch.sigmoid(model(x[start:start + batch_size])).tolist())
    return scores


def mean_loss(model, x, y, loss_function, batch_size):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            total += loss_function(model(x[start:start + batch_size]), y[start:start + batch_size]).item() * len(y[start:start + batch_size])
    return total / max(1, len(x))


def load_dataset():
    path = config.DATA_DIRECTORY / "dataset.json"
    if not path.is_file():
        raise OSError(f"Training data not found at {path}. Run generate_data.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_training(seed, epochs, checkpoint=None, shuffle_labels=False, quiet=False, data=None):
    """Train one CNN. Returns a results dict; saves weights only when `checkpoint` is given."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.NUM_THREADS)
    rng = random.Random(seed)
    data = data or load_dataset()
    encoder = PathEncoder()
    encode = lambda paths: torch.tensor([encoder.encode(p) for p in paths], dtype=torch.long)
    labels = {split: [r["label"] for r in records] for split, records in data.items()}
    train_labels = list(labels["train"])
    if shuffle_labels:  # Control run: a model that learns from labels must fail on shuffled ones.
        rng.shuffle(train_labels)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    x = {s: encode([r["path"] for r in data[s]]) for s in ("validation", "test")}
    y = {s: torch.tensor(labels[s], dtype=torch.float32) for s in ("validation", "test")}
    settings = model_settings()
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
    best_key = None  # Tuple compared as "larger is better"; see checkpoint_key below.
    by_auc = config.CHECKPOINT_BY == "auc"
    history, started, stopped_early = [], time.monotonic(), False
    if not quiet:
        ui.info(f"Training on {len(train_labels)} paths ({positives} positive), validating on {len(labels['validation'])}, "
                f"testing on {len(labels['test'])}; seed {seed}"
                + (" [SHUFFLED-LABEL CONTROL]" if shuffle_labels else ""))
        print(ui.style(f"{'epoch':>5}  {'':22}  {'train':>7}  {'valid':>7}  {'val AUC':>7}  {'val F1':>6}  {'lr':>8}", "bold"))
    try:
        for epoch in range(1, epochs + 1):
            paths = [r["path"] for r in data["train"]]
            if config.AUGMENT_TRAINING_PATHS:
                paths = [augment_path(p, rng) for p in paths]
            loader = DataLoader(TensorDataset(encode(paths), y_train), batch_size=config.BATCH_SIZE, shuffle=True)
            model.train()
            total = 0.0
            for batch_x, batch_y in loader:
                loss = loss_function(model(batch_x), batch_y)
                optimizer.zero_grad()
                loss.backward()  # Chain rule computes derivatives through CNN operations.
                if config.GRADIENT_CLIP:
                    nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
                optimizer.step()  # Update embedding, convolution and output weights.
                total += loss.item() * len(batch_y)
            validation_loss = mean_loss(model, x["validation"], y["validation"], loss_function, config.BATCH_SIZE)
            epoch_scores = predict_scores(model, x["validation"], config.BATCH_SIZE)
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
                best_key, best_loss, best_epoch, stale = key, validation_loss, epoch, 0
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
    # Threshold is chosen on validation data only; the test set is touched exactly once.
    validation_scores = predict_scores(model, x["validation"], config.BATCH_SIZE)
    tuned, tuned_score = tune_threshold(labels["validation"], validation_scores, config.THRESHOLD_F_BETA)
    threshold = tuned if config.AUTO_TUNE_THRESHOLD else config.PREDICTION_THRESHOLD
    test_scores = predict_scores(model, x["test"], config.BATCH_SIZE)
    results = metrics(labels["test"], [int(s >= threshold) for s in test_scores])
    at_default = metrics(labels["test"], [int(s >= 0.5) for s in test_scores])
    test_positive_rate = sum(labels["test"]) / max(1, len(labels["test"]))
    results.update(auc=roc_auc(labels["test"], test_scores), threshold=threshold,
                   f1_at_0_5=at_default["f1"], tuned_validation_fbeta=tuned_score,
                   test_loss=mean_loss(model, x["test"], y["test"], loss_function, config.BATCH_SIZE),
                   baseline_all_positive_f1=2 * test_positive_rate / (1 + test_positive_rate) if test_positive_rate else 0.0,
                   max_conv_weight_change=weight_change, history=history, test_examples=len(labels["test"]),
                   seed=seed, best_epoch=best_epoch, best_validation_loss=best_loss,
                   epochs_run=len(history), stopped_early=stopped_early, shuffled_labels=shuffle_labels,
                   seconds=time.monotonic() - started)
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), model_settings=settings,
                        encoder=encoder.settings(), threshold=float(threshold)), checkpoint)
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
    last = results["history"][-1]
    if last["validation"] > min(h["validation"] for h in results["history"]) * 1.1:
        ui.warn("Validation loss worsened after the best epoch: evidence of overfitting. The best earlier checkpoint was retained.")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the path CNN (defaults come from config.py)")
    parser.add_argument("--epochs", type=int, help="override EPOCHS")
    parser.add_argument("--seed", type=int, help="override RANDOM_SEED for this run")
    parser.add_argument("--seeds", type=int, metavar="N", help="repeat N seeds on the same data; saves nothing")
    parser.add_argument("--control", action="store_true", help="shuffled-label control; saves nothing")
    args = parser.parse_args()
    if args.seeds:
        seed_study(args.seeds, args.epochs)
    elif args.control:
        control(args.epochs)
    else:
        train(args.epochs, args.seed)
