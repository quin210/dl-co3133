import argparse
import copy
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from models import build

# every setting comes from configs/a3.yaml so the runs trace back to one file
CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "a3.yaml").read_text())
SHARED = CONFIG["shared"]
FEATURES = Path(__file__).resolve().parent.parent / "data" / "aokvqa" / "features"
OUTPUTS = Path(__file__).resolve().parent / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_features():
    meta = pd.read_pickle(FEATURES / "meta.pkl")
    image = torch.load(FEATURES / "image.pt")
    text = torch.load(FEATURES / "text.pt")
    # everything fits on one GPU, so keep it there and index batches directly
    data = {key: value.to(DEVICE) for key, value in {**image, **text}.items()}
    data["image_index"] = torch.tensor(meta["image_index"].values, device=DEVICE)
    data["label"] = torch.tensor(meta["label"].values, device=DEVICE)
    return meta, data

def make_batch(data, rows, image_rows=None):
    # rows are question indices; image_rows lets the shuffle test swap in other images
    images = data["image_index"][rows if image_rows is None else image_rows]
    return {"v": data["v"][images], "v_tokens": data["v_tokens"][images], "v_before_last": data["v_before_last"][images],
            "q": data["q"][rows], "c": data["c"][rows], "qc": data["qc"][rows],
            "qc_tokens": data["qc_tokens"][rows], "qc_mask": data["qc_mask"][rows]}

def da_score(meta, rows, predicted):
    # A-OKVQA direct-answer score of the chosen text: min(#workers who gave it / 3, 1)
    scores = []
    for row, choice in zip(rows, predicted):
        text = meta["choices"].iloc[row][choice].lower().strip()
        answers = [a.lower().strip() for a in meta["direct_answers"].iloc[row]]
        scores.append(min(answers.count(text) / 3, 1.0))
    return float(np.mean(scores))

@torch.no_grad()
def predict(model, data, rows, shuffle_seed=None):
    model.eval()
    image_rows = None
    if shuffle_seed is not None:
        # give every question the image of another question (ablation: does the model use the image?)
        generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
        image_rows = rows[torch.randperm(len(rows), generator=generator).to(rows.device)]
    scores = []
    for start in range(0, len(rows), 256):
        part = rows[start:start + 256]
        part_images = None if image_rows is None else image_rows[start:start + 256]
        scores.append(model(make_batch(data, part, part_images)).float())
    return torch.cat(scores)

def run_epoch(model, data, rows, optimizer):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    order = rows[torch.randperm(len(rows), device=rows.device)]
    for start in range(0, len(order), SHARED["batch_size"]):
        part = order[start:start + SHARED["batch_size"]]
        logits = model(make_batch(data, part))
        loss = nn.functional.cross_entropy(logits, data["label"][part])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(part)
        correct += (logits.argmax(1) == data["label"][part]).sum().item()
        total += len(part)
    return loss_sum / total, correct / total

def fit(name, seed, meta, data):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = build(CONFIG["runs"][name]["model"]).to(DEVICE)
    # the CLIP block (fine-tune ablation) gets a much smaller learning rate than the new heads
    block = [p for n, p in model.named_parameters() if n.startswith("block.")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("block.")]
    groups = [{"params": heads, "lr": SHARED["lr"]}]
    if block:
        groups.append({"params": block, "lr": SHARED["lr_clip_block"]})
    optimizer = torch.optim.Adam(groups)
    rows = {s: torch.tensor(np.where(meta["split"] == s)[0], device=DEVICE) for s in ("train", "validation", "test")}

    history, best_acc, best_state, best_epoch = [], -1.0, None, 0
    start = time.perf_counter()
    for epoch in range(1, SHARED["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, data, rows["train"], optimizer)
        val_logits = predict(model, data, rows["validation"])
        val_loss = nn.functional.cross_entropy(val_logits, data["label"][rows["validation"]]).item()
        val_acc = (val_logits.argmax(1) == data["label"][rows["validation"]]).float().mean().item()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc, "seconds": time.perf_counter() - epoch_start})
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, epoch, copy.deepcopy(model.state_dict())
        print("%s seed %d epoch %02d | loss %.4f/%.4f | accuracy train %.4f val %.4f" % (
            name, seed, epoch, train_loss, val_loss, train_acc, val_acc), flush=True)
    fit_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)

    # the test split is used once, after the checkpoint is chosen
    results = {}
    for split in ("validation", "test"):
        logits = predict(model, data, rows[split])
        predicted = logits.argmax(1)
        labels = data["label"][rows[split]]
        results[split] = {"accuracy": (predicted == labels).float().mean().item(),
                          "da_score": da_score(meta, rows[split].tolist(), predicted.tolist())}
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    timer = time.perf_counter()
    test_logits = predict(model, data, rows["test"])
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    results["test"]["head_ms_per_question"] = (time.perf_counter() - timer) * 1000 / len(rows["test"])
    shuffled = predict(model, data, rows["test"], shuffle_seed=seed).argmax(1)
    results["test"]["shuffled_image_accuracy"] = (shuffled == data["label"][rows["test"]]).float().mean().item()

    folder = OUTPUTS / ("%s_seed%d_%s" % (name, seed, datetime.now().strftime("%Y%m%d_%H%M%S")))
    folder.mkdir(parents=True)
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    np.savez_compressed(folder / "test_predictions.npz", rows=rows["test"].cpu().numpy(),
                        labels=data["label"][rows["test"]].cpu().numpy(),
                        scores=test_logits.cpu().numpy(), shuffled=shuffled.cpu().numpy())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    info = {"model": name, "seed": seed, "best_epoch": best_epoch, "fit_seconds": fit_seconds,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "validation": results["validation"], "test": results["test"], "settings": SHARED,
            "device": str(DEVICE), "gpu": torch.cuda.get_device_name() if DEVICE.type == "cuda" else None,
            "source_commit": commit, "source_uncommitted": bool(dirty),
            "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform()}}
    (folder / "metrics.json").write_text(json.dumps(info, indent=2) + "\n")
    torch.save({"model_state": best_state, "info": info}, folder / "best.pt")
    print("%s seed %d | best epoch %d | test accuracy %.4f DA %.4f | shuffled images %.4f | %s" % (
        name, seed, best_epoch, results["test"]["accuracy"], results["test"]["da_score"],
        results["test"]["shuffled_image_accuracy"], folder.name), flush=True)

def main():
    parser = argparse.ArgumentParser(description="Train the Assignment 3 models in configs/a3.yaml")
    parser.add_argument("models", nargs="*", help="run names (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=SHARED["seeds"])
    args = parser.parse_args()
    meta, data = load_features()
    for seed in args.seeds:
        for name in args.models or list(CONFIG["runs"]):
            fit(name, seed, meta, data)

if __name__ == "__main__":
    main()
