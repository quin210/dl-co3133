from io import BytesIO
from pathlib import Path
import ast
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent / "data" / "aokvqa" / "data"
MANIFEST = Path(__file__).resolve().parent / "splits.csv"
REVISION = "d1b0efa3a436e9101dfbde3752db7607da696c35"
SIZE = 224
SPLITS = ("train", "validation", "test")
# CLIP statistics, because the planned backbone is CLIP
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def read_split(split):
    files = sorted(ROOT.glob("%s-*.parquet" % split))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def read_aokvqa():
    # official test has no labels, so only train and validation are used
    parts = []
    for split in ("train", "validation"):
        part = read_split(split)
        part["source_split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    # the image is stored as a dict with the jpeg bytes, keep only the bytes
    df["image"] = [item["bytes"] for item in df["image"]]
    # direct_answers is stored as a string like "['a', 'b']", so turn it back into a list
    df["direct_answers"] = [ast.literal_eval(x) if isinstance(x, str) else list(x) for x in df["direct_answers"]]
    # no image id in the parquet, so the md5 of the bytes identifies an image
    df["image_key"] = [hashlib.md5(b).hexdigest() if b is not None else "missing" for b in df["image"]]
    return df

def shared_keys(df):
    # image keys that appear in both official splits
    train = set(df.loc[df["source_split"] == "train", "image_key"])
    val = set(df.loc[df["source_split"] == "validation", "image_key"])
    return train & val

def make_split(df, seed=42, val_ratio=0.1):
    shared = shared_keys(df)
    train_all = df[(df["source_split"] == "train") & ~df["image_key"].isin(shared)]
    # group by image so all questions of one image stay together
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_all, groups=train_all["image_key"]))
    parts = [train_all.iloc[train_idx].assign(split="train"),
             train_all.iloc[val_idx].assign(split="validation"),
             df[df["source_split"] == "validation"].assign(split="test")]
    manifest = pd.concat(parts)[["question_id", "image_key", "split"]].reset_index(drop=True)
    keys = {s: set(manifest.loc[manifest["split"] == s, "image_key"]) for s in SPLITS}
    assert not keys["train"] & keys["validation"]
    assert not keys["train"] & keys["test"]
    assert not keys["validation"] & keys["test"]
    manifest.to_csv(MANIFEST, index=False)
    return manifest

class AOKVQADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def load_image(self, data):
        image = Image.open(BytesIO(data)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        tensor = torch.from_numpy(np.asarray(image).copy())
        tensor = tensor.permute(2, 0, 1).float() / 255
        return (tensor - self.mean) / self.std

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return self.load_image(row["image"]), row["question"], list(row["choices"]), int(row["correct_choice_idx"])

def collate(batch):
    images, questions, choices, labels = zip(*batch)
    return torch.stack(images), list(questions), list(choices), torch.tensor(labels, dtype=torch.int64)

def get_data_loaders(batch_size=32, seed=42):
    df = read_aokvqa()
    manifest = make_split(df, seed)
    # one row per question id, the manifest tells which split it belongs to
    table = df.drop_duplicates("question_id").set_index("question_id")
    loaders = {}
    for split in SPLITS:
        ids = manifest.loc[manifest["split"] == split, "question_id"]
        data = AOKVQADataset(table.loc[ids].reset_index())
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(data, batch_size=batch_size, shuffle=(split == "train"), generator=generator,
                                    collate_fn=collate, num_workers=0)
    return loaders["train"], loaders["validation"], loaders["test"]

def main():
    train, validation, test = get_data_loaders()
    print("Train: %d | validation: %d | test: %d" % (len(train.dataset), len(validation.dataset), len(test.dataset)))
    images, questions, choices, labels = next(iter(train))
    print("Images: %s %s | labels: %s %s" % (tuple(images.shape), images.dtype, tuple(labels.shape), labels.dtype))
    print("Questions: %d strings | choices: %d lists of %d" % (len(questions), len(choices), len(choices[0])))
    print("Example: %s %s -> %s" % (questions[0], choices[0], choices[0][labels[0]]))

if __name__ == "__main__":
    main()
