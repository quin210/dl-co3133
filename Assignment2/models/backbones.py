from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights, convnext_tiny, efficientnet_b0

class EfficientNetB0(nn.Module):
    # ImageNet-pretrained EfficientNet-B0; the last linear layer is replaced by an eight-class head
    def __init__(self):
        super().__init__()
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, 8)

    def forward(self, images):
        return self.net(images)

class ConvNeXtTiny(nn.Module):
    # ImageNet-pretrained ConvNeXt-Tiny; the last linear layer is replaced by an eight-class head
    def __init__(self):
        super().__init__()
        self.net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.net.classifier[2] = nn.Linear(self.net.classifier[2].in_features, 8)

    def forward(self, images):
        return self.net(images)
