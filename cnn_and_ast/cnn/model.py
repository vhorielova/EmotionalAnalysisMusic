import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class EmotionCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.backbone(x)