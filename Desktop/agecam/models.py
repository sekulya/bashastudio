# models.py
import torch
from torchvision import models

class AgeClassifier(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = torch.nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )
    
    def forward(self, x):
        return self.backbone(x)