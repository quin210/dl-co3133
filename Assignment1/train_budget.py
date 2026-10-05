from pathlib import Path

import torch
from torch import nn

import train
from evaluate import evaluate


class MLPBudget(nn.Module): #MLP with 31 hidden units instead of 256 -> 24,655 parameters
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 31), nn.ReLU(), nn.Linear(31, 10))

    def forward(self, images):
        return self.layers(images)


class CNNBudget(nn.Module): #CNN, dense layer 13 instead of 64 -> 25,337 parameters
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 13), nn.ReLU(), nn.Linear(13, 10),
        )

    def forward(self, images):
        return self.layers(images)


class TransformerBudget(nn.Module): #Transformer, width 36 instead of 32 -> 24,274 parameters
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 36)
        self.positions = nn.Parameter(torch.empty(1, 49, 36))
        nn.init.normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=36, nhead=4, dim_feedforward=72, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = nn.Linear(36, 10)

    def forward(self, images):
        patches = images.unfold(2, 4, 4).unfold(3, 4, 4).reshape(images.shape[0], 49, 16)
        tokens = self.projection(patches) + self.positions
        return self.classifier(self.encoder(tokens).mean(dim=1))

#LSTM already has 24,714 parameters, so its runs in outputs are reused.

MODELS = [("MLPBudget25k", MLPBudget), ("CNNBudget25k", CNNBudget), ("TransformerBudget25k", TransformerBudget)]

if __name__ == "__main__":
    for name, model_class in MODELS:
        print(name, sum(p.numel() for p in model_class().parameters()), "parameters")

    train.OUTPUTS = Path(__file__).resolve().parent / "outputs_budget"
    for seed in (42, 43, 44, 45, 46):
        for name, model_class in MODELS:
            train.set_seed(seed) 
            folder = train.fit(model_class(), name, seed)
            evaluate(model_class(), folder)
