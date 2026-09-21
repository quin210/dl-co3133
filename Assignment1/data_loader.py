from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# paths stay the same when launched outside this folder
RAW = Path(__file__).resolve().parent.parent / "data" / "FashionMNIST" / "raw"
# labels follow the dataset's numeric order
NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def read_idx(path):
    with path.open("rb") as source:
        dimensions = source.read(4)[3]
        shape = np.fromfile(source, dtype=">u4", count=dimensions)
        return np.fromfile(source, dtype=np.uint8).reshape(tuple(shape))

def read_fashion_mnist():
    images = read_idx(RAW / "train-images-idx3-ubyte")
    labels = read_idx(RAW / "train-labels-idx1-ubyte")
    test_images = read_idx(RAW / "t10k-images-idx3-ubyte")
    test_labels = read_idx(RAW / "t10k-labels-idx1-ubyte")
    return images, labels, test_images, test_labels

def split_training_indices(labels):
    return train_test_split(np.arange(len(labels)), test_size=0.1, random_state=42, stratify=labels)

class FashionMNISTDataset(Dataset):
    # normalize once so epochs only index tensors
    def __init__(self, images, labels, mean, std):
        normalized = (images.astype(np.float32) / np.float32(255) - np.float32(mean)) / np.float32(std)
        self.images = torch.from_numpy(normalized).unsqueeze(1)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

def get_data_loaders(batch_size=128, seed=42, split=None, normalization=None):
    images, labels, test_images, test_labels = read_fashion_mnist()
    if split is None:
        train_indices, val_indices = split_training_indices(labels)
    else:
        train_indices = np.asarray(split["train_indices"])
        val_indices = np.asarray(split["val_indices"])

    if normalization is None:
        # validation and test pixels do not fit preprocessing
        counts = np.bincount(images[train_indices].reshape(-1), minlength=256)
        levels = np.arange(256, dtype=np.float64) / 255
        probability = counts / counts.sum()
        mean = float(probability @ levels)
        std = float(np.sqrt(probability @ ((levels - mean) ** 2)))
    else:
        mean, std = normalization["mean"], normalization["std"]

    dataset = FashionMNISTDataset(images, labels, mean, std)
    train = Subset(dataset, train_indices.tolist())
    validation = Subset(dataset, val_indices.tolist())
    test = FashionMNISTDataset(test_images, test_labels, mean, std)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    metadata = {"split_seed": 42, "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(), "mean": mean, "std": std}
    return train_loader, val_loader, test_loader, metadata


def main():
    train, validation, test, metadata = get_data_loaders()
    images, labels = next(iter(train))
    print("Train: %d | validation: %d | test: %d" % (len(train.dataset), len(validation.dataset), len(test.dataset)))
    print("Images: %s %s | labels: %s %s" % (tuple(images.shape), images.dtype, tuple(labels.shape), labels.dtype))
    print("Mean: %.6f | std: %.6f" % (metadata["mean"], metadata["std"]))

if __name__ == "__main__":
    main()
