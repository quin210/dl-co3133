from torch import nn

class MLP(nn.Module):
    # a nonlinear baseline without spatial assumptions
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images):
        return self.layers(images)
