from torch import nn

class CNN(nn.Module):
    # small baseline trained from scratch, no pretrained weights
    def __init__(self):
        super().__init__()
        def block(inputs, outputs, stride=1):
            return [nn.Conv2d(inputs, outputs, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(), nn.MaxPool2d(2)]
        # the first block uses stride 2 because 224 x 224 inputs make a full-resolution convolution slow on cpu
        self.layers = nn.Sequential(
            *block(3, 16, stride=2), *block(16, 32), *block(32, 64), *block(64, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
        )

    def forward(self, images):
        return self.layers(images)
