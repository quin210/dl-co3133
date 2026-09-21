from torch import nn

class LinearClassifier(nn.Module):
    # one score per class from the raw pixel vector
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def forward(self, images):
        return self.layers(images)
