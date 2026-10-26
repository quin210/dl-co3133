import numpy as np
import torch

from models.linear import LinearClassifier
from models.mlp import MLP
from models.cnn import CNN
from models.lstm import LSTMClassifier
from models.transformer import TransformerClassifier

# from data_loader import get_data_loaders
# train_loader, val_loader, test_loader, metadata = get_data_loaders()
# MEAN, STD = metadata["mean"], metadata["std"]
MEAN, STD = 0.28620900776977454, 0.3531597432932429 # hardcoded for quicker run

# same float32 arithmetic as the dataset, so the background is exactly a normalized 0 pixel
LOW = float((np.float32(0) - np.float32(MEAN)) / np.float32(STD))
HIGH = float((np.float32(1) - np.float32(MEAN)) / np.float32(STD))
SIGMA = 0.10 / STD
MAX_SHIFT = 2

def augment(images):
    # random shift in {-2..2} pixels per axis, then noise sigma 0.10 with probability 0.5, per image
    batch, _, height, width = images.shape
    padded = torch.nn.functional.pad(images, (MAX_SHIFT,) * 4, value=LOW)
    dy = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (batch,))
    dx = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (batch,))
    rows = (MAX_SHIFT - dy)[:, None] + torch.arange(height)
    cols = (MAX_SHIFT - dx)[:, None] + torch.arange(width)
    shifted = padded[torch.arange(batch)[:, None, None], 0, rows[:, :, None], cols[:, None, :]].unsqueeze(1)
    noisy = torch.rand(batch, device=images.device) < 0.5
    noise = torch.randn_like(shifted) * SIGMA * noisy[:, None, None, None]
    return torch.where(noisy[:, None, None, None], (shifted + noise).clamp(LOW, HIGH), shifted)

# each class augments only in training mode; evaluation sees clean images like the base model

class AugmentedLinear(LinearClassifier):
    def forward(self, images):
        if self.training:
            images = augment(images)
        return super().forward(images)

class AugmentedMLP(MLP):
    def forward(self, images):
        if self.training:
            images = augment(images)
        return super().forward(images)

class AugmentedCNN(CNN):
    def forward(self, images):
        if self.training:
            images = augment(images)
        return super().forward(images)

class AugmentedLSTM(LSTMClassifier):
    def forward(self, images):
        if self.training:
            images = augment(images)
        return super().forward(images)

class AugmentedTransformer(TransformerClassifier):
    def forward(self, images):
        if self.training:
            images = augment(images)
        return super().forward(images)
