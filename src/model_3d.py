"""
3D CNN models for brain MRI classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block."""
    
    def __init__(self, in_channels:  int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class Simple3DCNN(nn.Module):
    """Simple 3D CNN for brain MRI classification."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_filters: int = 32,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1: 128 -> 64
            ConvBlock3D(in_channels, base_filters),
            nn.MaxPool3d(2),
            
            # Block 2: 64 -> 32
            ConvBlock3D(base_filters, base_filters * 2),
            nn.MaxPool3d(2),
            
            # Block 3: 32 -> 16
            ConvBlock3D(base_filters * 2, base_filters * 4),
            nn.MaxPool3d(2),
            
            # Block 4: 16 -> 8
            ConvBlock3D(base_filters * 4, base_filters * 8),
            nn.MaxPool3d(2),
            
            # Block 5: 8 -> 4
            ConvBlock3D(base_filters * 8, base_filters * 16),
            nn.AdaptiveAvgPool3d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_filters * 16, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class ResBlock3D(nn.Module):
    """3D Residual block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn. Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self. conv1(x)))
        out = self.bn2(self. conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    """3D ResNet for brain MRI classification."""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        layers: list = [2, 2, 2, 2],
        base_filters: int = 64,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.in_channels = base_filters
        
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters, base_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)
        
        self. avgpool = nn.AdaptiveAvgPool3d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_filters * 8, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResBlock3D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.classifier(x)


def get_3d_model(
    model_type: str = 'simple',
    num_classes: int = 4,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Factory function for 3D models."""
    
    if model_type == 'simple':
        model = Simple3DCNN(num_classes=num_classes)
    elif model_type == 'resnet':
        model = ResNet3D(num_classes=num_classes, layers=[2, 2, 2, 2])
    elif model_type == 'resnet_small':
        model = ResNet3D(num_classes=num_classes, layers=[1, 1, 1, 1], base_filters=32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if device: 
        model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[3D Model] {model_type}: {total_params: ,} parameters")
    
    return model