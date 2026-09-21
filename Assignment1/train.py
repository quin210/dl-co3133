import copy
import csv
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from data_loader import get_data_loaders, read_fashion_mnist
from models.linear import LinearClassifier
from models.mlp import MLP
from models.cnn import CNN
from models.lstm import LSTMClassifier
from models.transformer import TransformerClassifier

# the same budget and selection rule apply to every architecture
BATCH_SIZE = 128
EPOCHS = 16
LEARNING_RATE = 0.001
CLIP_NORM = 5.0
SEEDS = (42,)
# the same cpu thread count applies to every model
THREADS = 4
DEVICE = torch.device("cpu")
OUTPUTS = Path(__file__).resolve().parent / "outputs"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)

def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions = [], []
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    metrics = {
        "loss": total_loss / len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=range(10), average="macro", zero_division=0)),
    }
    return metrics, np.asarray(labels), np.asarray(predictions)

def save_curves(history, folder, name, best_epoch):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, metric in zip(axes, ["loss", "accuracy", "macro_f1"]):
        ax.plot(epochs, [row["train_" + metric] for row in history], "--", label="Training")
        ax.plot(epochs, [row["val_" + metric] for row in history], label="Validation")
        ax.axvline(best_epoch, color="gray", linestyle=":", label="Selected epoch")
        ax.set(title=metric.replace("_", " ").title(), xlabel="Epoch")
        ax.set_xticks(epochs)
        if metric != "loss":
            ax.set_ylim(0, 1)
    axes[0].legend()
    fig.suptitle(name)
    fig.savefig(folder / "learning_curves.png", dpi=150)
    plt.close(fig)

def save_validation(model, loader, indices, folder):
    model.eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for images, targets in loader:
            probabilities.append(model(images.to(DEVICE)).softmax(1).cpu().numpy())
            labels.append(targets.numpy())
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    raw_images, _, _, _ = read_fashion_mnist()
    np.savez_compressed(folder / "validation_predictions.npz", indices=indices,
                        images=raw_images[indices], labels=labels,
                        predictions=probabilities.argmax(1), probabilities=probabilities)

def fit(model, name, seed=42):
    train_loader, val_loader, _, data = get_data_loaders(BATCH_SIZE, seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    folder = OUTPUTS / ("%s_seed%d_%s" % (name, seed, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    folder.mkdir(parents=True)
    history, best_state, best_f1, best_epoch = [], None, -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, optimizer)
        val_metrics, _, _ = run_epoch(model, val_loader)
        row = {"epoch": epoch}
        row.update({"train_" + key: value for key, value in train_metrics.items()})
        row.update({"val_" + key: value for key, value in val_metrics.items()})
        history.append(row)
        with (folder / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
        print("%s seed %d epoch %02d | loss %.4f/%.4f | accuracy train %.4f val %.4f | val F1 %.4f" % (
            name, seed, epoch, train_metrics["loss"], val_metrics["loss"],
            train_metrics["accuracy"], val_metrics["accuracy"], val_metrics["macro_f1"]), flush=True)
    fit_seconds = time.perf_counter() - start
    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    fixed_train = DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_metrics, _, _ = run_epoch(model, fixed_train)
    val_metrics, _, _ = run_epoch(model, val_loader)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent,
                           capture_output=True, text=True).stdout.strip()
    info = {
        "model": name, "model_class": type(model).__name__, "seed": seed,
        "best_epoch": best_epoch, "epochs_run": epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fit_seconds": fit_seconds, "train": train_metrics, "validation": val_metrics,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "max_epochs": EPOCHS, "clip_norm": CLIP_NORM,
        "optimizer": "Adam", "checkpoint_rule": "highest validation macro-F1; earliest tie",
        "augmentation": None, "scheduler": None, "weight_decay": 0.0,
        "device": str(DEVICE), "threads": THREADS, "source_commit": commit,
        "source_uncommitted": bool(dirty),
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "timing": "epoch training, validation, history writes and best-weight copies",
    }
    torch.save({"model_state": best_state, "epoch": best_epoch, "info": info, "data": data}, folder / "best.pt")
    torch.save({"model_state": last_state, "epoch": epoch, "info": info, "data": data}, folder / "checkpoint.pt")
    (folder / "checkpoint_info.json").write_text(json.dumps(info, indent=2) + "\n")
    save_curves(history, folder, name, best_epoch)
    save_validation(model, val_loader, data["val_indices"], folder)
    print("Saved %s" % folder, flush=True)
    return folder

def main():
    for seed in SEEDS:
        set_seed(seed)
        fit(LinearClassifier(), "Linear", seed)
        set_seed(seed)
        fit(MLP(), "MLP", seed)
        set_seed(seed)
        fit(CNN(), "CNN", seed)
        set_seed(seed)
        fit(LSTMClassifier(), "LSTM", seed)
        set_seed(seed)
        fit(TransformerClassifier(), "Transformer", seed)

if __name__ == "__main__":
    main()
