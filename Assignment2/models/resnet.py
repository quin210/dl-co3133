from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

class ResNet18(nn.Module):
    # ImageNet-pretrained backbone with a new eight-class head
    def __init__(self, freeze_backbone):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(self.net.fc.in_features, 8)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, parameter in self.net.named_parameters():
                parameter.requires_grad = name.startswith("fc.")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            # frozen layers must also keep their BatchNorm statistics
            for name, module in self.net.named_children():
                if name != "fc":
                    module.eval()
        return self

    def forward(self, images):
        return self.net(images)
