import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from data_loader import get_data_loaders
from train import DEVICE, OUTPUTS, RUNS, THREADS

def predict(model, loader):
    labels, probabilities = [], []
    loss = 0.0
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(DEVICE))
            loss += torch.nn.functional.cross_entropy(logits, targets.to(DEVICE), reduction="sum").item()
            labels.append(targets.numpy())
            probabilities.append(logits.softmax(1).cpu().numpy())
    labels, probabilities = np.concatenate(labels), np.concatenate(probabilities)
    predictions = probabilities.argmax(1)
    metrics = {
        "loss": loss / len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=range(8), average="macro", zero_division=0)),
    }
    return metrics, labels, predictions, probabilities

def inference_time(model, loader):
    # one warmup, then the median of two complete passes including batch loading
    durations = []
    with torch.inference_mode():
        for repeat in range(3):
            start = time.perf_counter()
            for images, _ in loader:
                model(images.to(DEVICE)).argmax(1)
            if repeat:
                durations.append(time.perf_counter() - start)
    return float(np.median(durations) * 1_000 / len(loader.dataset))

def evaluate(name, folder):
    folder = Path(folder)
    if (folder / "metrics.json").exists():
        print("Skip (already evaluated):", folder)
        return None
    torch.set_num_threads(THREADS)
    saved = torch.load(folder / "best.pt", map_location=DEVICE, weights_only=True)
    info = saved["info"]
    model = RUNS[name]["build"]().to(DEVICE)
    if type(model).__name__ != info["model_class"]:
        raise ValueError("model class differs from checkpoint")
    model.load_state_dict(saved["model_state"])
    _, val_loader, test_loader, _ = get_data_loaders(info["batch_size"], info["seed"])
    results = dict(info)
    for split, loader in [("validation", val_loader), ("test", test_loader)]:
        metrics, labels, predictions, probabilities = predict(model, loader)
        metrics["inference_ms"] = inference_time(model, loader)
        results[split] = metrics
        np.savez_compressed(folder / (split + "_predictions.npz"), labels=labels,
                            predictions=predictions, probabilities=probabilities)
        print("%s %s | accuracy %.4f F1 %.4f | %.2f ms/image" % (
            info["model"], split, metrics["accuracy"], metrics["macro_f1"], metrics["inference_ms"]), flush=True)
    results["inference_timing"] = "median of 2 full passes after 1 warmup; DataLoader + forward + argmax; no metrics"
    (folder / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results

def main():
    for name in sys.argv[1:] or list(RUNS):
        for folder in sorted(OUTPUTS.glob(name + "_seed*/best.pt")):
            evaluate(name, folder.parent)

if __name__ == "__main__":
    main()
