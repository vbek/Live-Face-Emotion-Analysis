import torch
import torch.nn as nn
from torchvision import models

class ResNet18Emotion(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, freeze_until_layer="layer1", dropout_p=0.3):
        
        super().__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Freeze early layers
        if freeze_until_layer is not None:
            freeze = True
            for name, param in self.model.named_parameters():
                if freeze:
                    param.requires_grad = False
                if freeze_until_layer in name:
                    freeze = False
        
        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
