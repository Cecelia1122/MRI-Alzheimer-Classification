"""
Model architectures for Alzheimer's MRI classification. 
Supports both 2D and 3D models.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class AlzheimerClassifier2D(nn.Module):
    """2D CNN classifier with transfer learning."""
    
    SUPPORTED_BACKBONES = ['resnet50', 'resnet18', 'efficientnet_b0', 'densenet121']
    
    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone must be one of {self.SUPPORTED_BACKBONES}")
        
        self.backbone_name = backbone
        
        if backbone == 'resnet50': 
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn. Identity()
        elif backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone. fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet_b0': 
            weights = models.EfficientNet_B0_Weights. IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'densenet121': 
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            num_features = self.backbone.classifier.in_features
            self.backbone. classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
        
        self.num_features = num_features
    
    def forward(self, x:  torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze


def get_model(
    backbone: str = 'resnet50',
    num_classes: int = 4,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    model_dim: str = '2d'
) -> nn.Module:
    """Factory function to create model."""
    
    if model_dim == '2d': 
        model = AlzheimerClassifier2D(num_classes, backbone, pretrained)
    elif model_dim == '3d':
        from . model_3d import get_3d_model
        model = get_3d_model(model_type=backbone, num_classes=num_classes, device=None)
    else:
        raise ValueError(f"model_dim must be '2d' or '3d', got {model_dim}")
    
    if device:
        model = model. to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {backbone} ({model_dim}) - {total_params:,} parameters")
    
    return model