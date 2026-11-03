from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent / "data" / "tomato"
RAW = ROOT / "raw"
CACHE = ROOT / "images_224.npz"
SIZE = 224
# labels follow the sorted folder names, as in the M1 proposal
NAMES = ["Bacterial spot", "Early blight", "Late blight", "Leaf mold", "Septoria leaf spot",
         "Tomato yellow leaf curl virus", "Tomato mosaic virus", "Healthy"]
FOLDERS = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
           "Tomato___Septoria_leaf_spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
           "Tomato___healthy"]
SPLITS = ("train", "validation", "test")
# ImageNet statistics, because the pretrained backbone expects them
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def list_files():
    paths, labels, splits = [], [], []
    for split_id, split in enumerate(SPLITS):
        for label, folder in enumerate(FOLDERS):
            for path in sorted((RAW / split / folder).iterdir()):
                paths.append(str(path.relative_to(RAW)))
                labels.append(label)
                splits.append(split_id)
    return np.array(paths), np.array(labels), np.array(splits)

def load_image(path):
    with Image.open(RAW / path) as image:
        return np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))

def read_tomato():
    # decode and resize once so later runs only read one array
    if not CACHE.exists():
        paths, labels, splits = list_files()
        images = np.stack([load_image(path) for path in paths])
        np.savez(CACHE, images=images, labels=labels, splits=splits, paths=paths)
    cached = np.load(CACHE)
    return cached["images"], cached["labels"], cached["splits"], cached["paths"]

# augmentation presets; "strong" is the A2 ablation and is described in configs/a2.yaml
AUGMENTATIONS = ("none", "flip", "strong")
STRONG = v2.Compose([
    v2.RandomResizedCrop(SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33), antialias=True),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
])

def check_augment(augment):
    # older code passes True / False; map them to the presets
    if augment is True:
        return "flip"
    if augment is False or augment is None:
        return "none"
    if augment not in AUGMENTATIONS:
        raise ValueError("augment must be one of %s, got %r" % (AUGMENTATIONS, augment))
    return augment

class TomatoDataset(Dataset):
    # keep uint8 pixels and normalize per batch to save memory
    def __init__(self, images, labels, augment="none"):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        # train.py switches augmentation off with augment = False before the final train metrics
        self._augment = check_augment(value)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment in ("flip", "strong"):
            if torch.rand(1) < 0.5:
                image = image.flip(2)
            if torch.rand(1) < 0.5:
                image = image.flip(1)
        if self.augment == "strong":
            # operates on uint8 pixels before normalisation; randomness comes from the seeded torch RNG
            image = STRONG(image)
        image = (image.float() / 255 - self.mean) / self.std
        return image, self.labels[index]

def get_data_loaders(batch_size=64, seed=42, augment="flip"):
    images, labels, splits, _ = read_tomato()
    parts = [splits == split_id for split_id in range(3)]
    train = TomatoDataset(images[parts[0]], labels[parts[0]], augment)
    validation = TomatoDataset(images[parts[1]], labels[parts[1]])
    test = TomatoDataset(images[parts[2]], labels[parts[2]])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    metadata = {"split": "as published in the dataset repository", "train": len(train), "validation": len(validation),
                "test": len(test), "mean": list(MEAN), "std": list(STD), "size": SIZE, "augment": train.augment}
    return train_loader, val_loader, test_loader, metadata

def main():
    train, validation, test, metadata = get_data_loaders()
    images, labels = next(iter(train))
    print("Train: %d | validation: %d | test: %d" % (len(train.dataset), len(validation.dataset), len(test.dataset)))
    print("Images: %s %s | labels: %s %s" % (tuple(images.shape), images.dtype, tuple(labels.shape), labels.dtype))

if __name__ == "__main__":
    main()