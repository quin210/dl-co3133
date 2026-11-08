import argparse
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
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from data_loader import get_data_loaders
from models.cnn import CNN
from models.backbones import ConvNeXtTiny, EfficientNetB0
from models.resnet import ResNet18

# every hyperparameter comes from configs/a2.yaml so reported runs trace back to one committed file
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "a2.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SHARED = CONFIG["shared"]
BATCH_SIZE = SHARED["batch_size"]
CLIP_NORM = SHARED["clip_norm"]
SEEDS = tuple(SHARED["seeds"])
THREADS = SHARED["threads"]
DEVICE = torch.device(SHARED["device"])
OUTPUTS = Path(__file__).resolve().parent / "outputs"
SMOKE_OUTPUTS = Path(__file__).resolve().parent / "outputs_smoke"

def builder(settings):
    if settings["model"] == "CNN":
        return lambda: CNN()
    if settings["model"] == "ResNet18":
        return lambda: ResNet18(freeze_backbone=settings["freeze_backbone"])
    if settings["model"] == "EfficientNetB0":
        return lambda: EfficientNetB0()
    if settings["model"] == "ConvNeXtTiny":
        return lambda: ConvNeXtTiny()
    raise ValueError("unknown model %r" % settings["model"])

# epochs and learning rate differ because the models start from different weights
RUNS = {name: dict(settings, build=builder(settings)) for name, settings in CONFIG["runs"].items()}

def device_of(name):
    # a run may set its own device; the CPU runs of the main study use the shared one
    return torch.device(RUNS[name].get("device", SHARED["device"]))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    # avoid autotuned convolution kernels on the GPU, so repeated runs pick the same algorithms
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_epoch(model, loader, optimizer=None, log_every=0, max_steps=0, device=DEVICE):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions = [], []
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader, 1):
            if max_steps and step > max_steps:
                break
            images, targets = images.to(device), targets.to(device)
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

def fit(name, seed=42, smoke=False):
    settings = RUNS[name]
    # smoke test: 1 epoch of a few batches, written to a separate folder that evaluate.py never reads
    epochs = 1 if smoke else settings["epochs"]
    max_steps = 5 if smoke else 0
    device = device_of(name)
    set_seed(seed)
    model = settings["build"]().to(device)
    train_loader, val_loader, _, data = get_data_loaders(BATCH_SIZE, seed, augment=settings["augment"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=settings["lr"])
    root = SMOKE_OUTPUTS if smoke else OUTPUTS
    folder = root / ("%s_seed%d_%s" % (name, seed, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    folder.mkdir(parents=True)
    history, best_state, best_f1, best_epoch = [], None, -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics, _, _ = run_epoch(model, train_loader, optimizer, log_every=50, max_steps=max_steps, device=device)
        val_metrics, _, _ = run_epoch(model, val_loader, max_steps=max_steps, device=device)
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
    train_metrics, _, _ = run_epoch(model, fixed_train, max_steps=max_steps, device=device)
    val_metrics, _, _ = run_epoch(model, val_loader, max_steps=max_steps, device=device)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent,
                           capture_output=True, text=True).stdout.strip()
    info = {
        "model": name, "model_class": type(model).__name__, "seed": seed,
        "best_epoch": best_epoch, "epochs_run": epochs, "smoke_test": smoke,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "fit_seconds": fit_seconds, "train": train_metrics, "validation": val_metrics,
        "batch_size": BATCH_SIZE, "learning_rate": settings["lr"],
        "max_epochs": settings["epochs"], "clip_norm": CLIP_NORM,
        "optimizer": SHARED["optimizer"], "checkpoint_rule": SHARED["checkpoint_rule"],
        "augment": settings["augment"], "augmentation": CONFIG["augmentations"][settings["augment"]],
        "scheduler": SHARED["scheduler"], "weight_decay": SHARED["weight_decay"],
        "pretrained": settings["pretrained"], "config_file": "configs/a2.yaml",
        "device": str(device), "threads": THREADS,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "source_commit": commit,
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
    parser = argparse.ArgumentParser(description="Train the Assignment 2 models listed in configs/a2.yaml")
    parser.add_argument("models", nargs="*", help="run names from configs/a2.yaml (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="override the seeds in the config")
    parser.add_argument("--smoke", action="store_true", help="1 epoch of 5 batches per model, saved to outputs_smoke/")
    args = parser.parse_args()
    names = args.models or list(RUNS)
    unknown = [name for name in names if name not in RUNS]
    if unknown:
        parser.error("unknown run(s) %s; choose from %s" % (unknown, list(RUNS)))
    for seed in args.seeds:
        for name in names:
            fit(name, seed, smoke=args.smoke)

if __name__ == "__main__":
    main()