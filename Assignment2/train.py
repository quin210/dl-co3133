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

from data_loader import get_data_loaders
from models.cnn import CNN
from models.resnet import ResNet18

# the same batch size, seed and selection rule apply to every model
BATCH_SIZE = 64
CLIP_NORM = 5.0
# M2 draft uses one seed; more seeds are added for the final report
SEEDS = (42,)
THREADS = 16
DEVICE = torch.device("cpu")
OUTPUTS = Path(__file__).resolve().parent / "outputs"
# epochs and learning rate differ because the models start from different weights
RUNS = {
    "CNN": {"build": lambda: CNN(), "epochs": 12, "lr": 1e-3},
    "ResNet18Frozen": {"build": lambda: ResNet18(freeze_backbone=True), "epochs": 6, "lr": 1e-3},
    "ResNet18FineTune": {"build": lambda: ResNet18(freeze_backbone=False), "epochs": 4, "lr": 1e-4},
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)

def run_epoch(model, loader, optimizer=None, log_every=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions = [], []
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader, 1):
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
            if log_every and step % log_every == 0:
                print("  step %d/%d | loss %.4f" % (step, len(loader), loss.item()), flush=True)
    metrics = {
        "loss": total_loss / len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=range(8), average="macro", zero_division=0)),
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

def fit(name, seed=42):
    settings = RUNS[name]
    set_seed(seed)
    model = settings["build"]().to(DEVICE)
    train_loader, val_loader, _, data = get_data_loaders(BATCH_SIZE, seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=settings["lr"])
    folder = OUTPUTS / ("%s_seed%d_%s" % (name, seed, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    folder.mkdir(parents=True)
    history, best_state, best_f1, best_epoch = [], None, -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_metrics, _, _ = run_epoch(model, train_loader, optimizer, log_every=50)
        val_metrics, _, _ = run_epoch(model, val_loader)
        row = {"epoch": epoch}
        row.update({"train_" + key: value for key, value in train_metrics.items()})
        row.update({"val_" + key: value for key, value in val_metrics.items()})
        row["seconds"] = time.perf_counter() - epoch_start
        history.append(row)
        with (folder / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
        print("%s seed %d epoch %02d | loss %.4f/%.4f | accuracy train %.4f val %.4f | val F1 %.4f | %.0fs" % (
            name, seed, epoch, train_metrics["loss"], val_metrics["loss"],
            train_metrics["accuracy"], val_metrics["accuracy"], val_metrics["macro_f1"], row["seconds"]), flush=True)
    fit_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    # final train metrics use the best weights without augmentation
    train_loader.dataset.augment = False
    fixed_train = DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_metrics, _, _ = run_epoch(model, fixed_train)
    val_metrics, _, _ = run_epoch(model, val_loader)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent,
                           capture_output=True, text=True).stdout.strip()
    info = {
        "model": name, "model_class": type(model).__name__, "seed": seed,
        "best_epoch": best_epoch, "epochs_run": settings["epochs"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "fit_seconds": fit_seconds, "train": train_metrics, "validation": val_metrics,
        "batch_size": BATCH_SIZE, "learning_rate": settings["lr"],
        "max_epochs": settings["epochs"], "clip_norm": CLIP_NORM,
        "optimizer": "Adam", "checkpoint_rule": "highest validation macro-F1; earliest tie",
        "augmentation": "random horizontal and vertical flip", "scheduler": None, "weight_decay": 0.0,
        "pretrained": "ImageNet1K_V1" if name.startswith("ResNet") else None,
        "device": str(DEVICE), "threads": THREADS, "source_commit": commit,
        "source_uncommitted": bool(dirty),
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "platform": platform.platform(), "machine": platform.machine()},
        "timing": "epoch training, validation, history writes and best-weight copies",
    }
    torch.save({"model_state": best_state, "epoch": best_epoch, "info": info, "data": data}, folder / "best.pt")
    (folder / "checkpoint_info.json").write_text(json.dumps(info, indent=2) + "\n")
    save_curves(history, folder, name, best_epoch)
    print("Saved %s" % folder, flush=True)
    return folder

def main():
    names = sys.argv[1:] or list(RUNS)
    for seed in SEEDS:
        for name in names:
            fit(name, seed)

if __name__ == "__main__":
    main()
