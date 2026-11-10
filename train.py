"""CPU training, validation checkpoint selection, and held-out evaluation."""
import json
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import config
from model import PathCNN, model_settings
from preprocessing import PathEncoder


def metrics(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return dict(precision=precision, recall=recall, f1=2 * precision * recall / max(1e-12, precision + recall))


def train():
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    torch.set_num_threads(2)
    data = json.loads((config.DATA_DIRECTORY / "dataset.json").read_text(encoding="utf-8"))
    encoder = PathEncoder()
    loaders = {}
    for split, records in data.items():
        x = torch.tensor([encoder.encode(r["path"]) for r in records], dtype=torch.long)
        y = torch.tensor([r["label"] for r in records], dtype=torch.float32)
        loaders[split] = DataLoader(TensorDataset(x, y), batch_size=config.BATCH_SIZE, shuffle=split == "train")
    settings = model_settings()
    model = PathCNN(**settings)
    initial = model.conv.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Binary cross-entropy penalizes wrong binary predictions; logits version is numerically stable.
    loss_function = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_epoch = 0
    history = []
    config.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.EPOCHS):
        losses = {}
        for split in ("train", "validation"):
            model.train(split == "train")
            total = 0.0
            with torch.set_grad_enabled(split == "train"):
                for x, y in loaders[split]:
                    loss = loss_function(model(x), y)
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()  # Chain rule computes derivatives through CNN operations.
                        optimizer.step()  # Update embedding, convolution and output weights.
                    total += loss.item() * len(y)
            losses[split] = total / len(loaders[split].dataset)
        history.append(losses)
        print(f"Epoch {epoch + 1:02d}: training loss={losses['train']:.4f}, validation loss={losses['validation']:.4f}")
        if losses["validation"] < best_loss:
            best_loss = losses["validation"]
            best_epoch = epoch + 1
            torch.save(dict(state_dict=model.state_dict(), model_settings=settings,
                            encoder=encoder.settings()), config.OUTPUT_DIRECTORY / "model.pt")
    weight_change = (initial - model.conv.weight.detach()).abs().max().item()
    checkpoint = torch.load(config.OUTPUT_DIRECTORY / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    labels, predicted = [], []
    with torch.no_grad():
        for x, y in loaders["test"]:
            labels.extend(y.tolist())
            predicted.extend((torch.sigmoid(model(x)) >= config.PREDICTION_THRESHOLD).int().tolist())
    results = metrics(labels, predicted)
    results.update(max_conv_weight_change=weight_change, history=history,
                   test_examples=len(labels), seed=config.RANDOM_SEED,
                   best_epoch=best_epoch, best_validation_loss=best_loss)
    (config.OUTPUT_DIRECTORY / "training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Held-out test:", {k: round(results[k], 4) for k in ("precision", "recall", "f1")})
    print(f"Maximum convolution weight change: {weight_change:.6f}")
    print(f"Selected epoch {best_epoch}, validation loss {best_loss:.4f}")
    if history[-1]["validation"] > best_loss * 1.1:
        print("Validation loss worsened: evidence of overfitting. The best earlier checkpoint was retained.")
    return results


if __name__ == "__main__":
    train()
