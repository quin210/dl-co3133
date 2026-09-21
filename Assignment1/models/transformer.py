import torch
from torch import nn

class TransformerClassifier(nn.Module):
    # attention combines patches with learned spatial positions
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 32)
        self.positions = nn.Parameter(torch.empty(1, 49, 32))
        nn.init.normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.classifier = nn.Linear(32, 10)

    def forward(self, images):
        patches = images.unfold(2, 4, 4).unfold(3, 4, 4)
        patches = patches.reshape(images.shape[0], 49, 16)
        tokens = self.projection(patches) + self.positions
        return self.classifier(self.encoder(tokens).mean(dim=1))
