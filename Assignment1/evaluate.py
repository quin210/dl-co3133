import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from data_loader import get_data_loaders, read_fashion_mnist
from models.linear import LinearClassifier
from models.mlp import MLP
from models.cnn import CNN
from models.lstm import LSTMClassifier
from models.transformer import TransformerClassifier
from train import DEVICE, OUTPUTS, THREADS

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
        "macro_f1": float(f1_score(labels, predictions, labels=range(10), average="macro", zero_division=0)),
    }
    return metrics, labels, predictions, probabilities

def inference_time(model, loader):
    # one warmup, then the median of three complete passes including batch loading
    durations = []
    with torch.inference_mode():
        for repeat in range(4):
            start = time.perf_counter()
            for images, _ in loader:
                model(images.to(DEVICE)).argmax(1)
            if repeat:
                durations.append(time.perf_counter() - start)
    return float(np.median(durations) * 1_000 / len(loader.dataset))

def evaluate(model, folder):
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    saved = torch.load(folder / "best.pt", map_location=DEVICE, weights_only=True)
    info, data = saved["info"], saved["data"]
    if type(model).__name__ != info["model_class"]:
        raise ValueError("model class differs from checkpoint")
    model = model.to(DEVICE)
    model.load_state_dict(saved["model_state"])
    _, val_loader, test_loader, _ = get_data_loaders(info["batch_size"], info["seed"], data, data)
    results = dict(info)
    raw_images, _, _, _ = read_fashion_mnist()
    for name, loader, indices in [("validation", val_loader, data["val_indices"]),
                                  ("test", test_loader, np.arange(len(test_loader.dataset)))]:
        metrics, labels, predictions, probabilities = predict(model, loader)
        metrics["inference_ms"] = inference_time(model, loader)
        results[name] = metrics
        saved_images = {"images": raw_images[indices]} if name == "validation" else {}
        np.savez_compressed(folder / (name + "_predictions.npz"), indices=indices,
                            labels=labels, predictions=predictions, probabilities=probabilities, **saved_images)
        print("%s %s | accuracy %.4f F1 %.4f | %.4f ms/image" % (
            info["model"], name, metrics["accuracy"], metrics["macro_f1"], metrics["inference_ms"]), flush=True)
    results["inference_timing"] = "median of 3 full passes after 1 warmup; DataLoader + forward + argmax; no metrics"
    (folder / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results

def main():
    for folder in sorted(OUTPUTS.glob("Linear_seed*/best.pt")):
        evaluate(LinearClassifier(), folder.parent)
    for folder in sorted(OUTPUTS.glob("MLP_seed*/best.pt")):
        evaluate(MLP(), folder.parent)
    for folder in sorted(OUTPUTS.glob("CNN_seed*/best.pt")):
        evaluate(CNN(), folder.parent)
    for folder in sorted(OUTPUTS.glob("LSTM_seed*/best.pt")):
        evaluate(LSTMClassifier(), folder.parent)
    for folder in sorted(OUTPUTS.glob("Transformer_seed*/best.pt")):
        evaluate(TransformerClassifier(), folder.parent)

if __name__ == "__main__":
    main()
