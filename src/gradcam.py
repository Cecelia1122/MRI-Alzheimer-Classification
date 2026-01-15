"""
Grad-CAM visualization. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple, List, Dict
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class GradCAM:
    """Grad-CAM visualization."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor. requires_grad = True
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output. backward(gradient=one_hot)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    
    def visualize(self, image_path: str, save_path: Optional[str] = None) -> plt.Figure:
        CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        cam = self.generate_cam(input_tensor)
        
        with torch.no_grad():
            output = self.model(input_tensor. to(self.device))
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)
        
        from scipy import ndimage
        cam_resized = ndimage.zoom(cam, (224 / cam.shape[0], 224 / cam.shape[1]), order=1)
        heatmap = plt.cm.jet(cam_resized)[:, :, :3] * 255
        heatmap = heatmap.astype(np. uint8)
        overlay = (heatmap * 0.4 + image_array * 0.6).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_array)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title(f'{CLASS_NAMES[pred_class]}\n{confidence:.1%}')
        axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig


def get_target_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    if 'resnet' in backbone_name:
        return model.backbone.layer4[-1]
    elif 'efficientnet' in backbone_name: 
        return model.backbone.features[-1]
    elif 'densenet' in backbone_name: 
        return model.backbone.features. denseblock4
    raise ValueError(f"Unknown backbone: {backbone_name}")


def visualize_samples_with_gradcam(
    model: nn.Module,
    backbone_name: str,
    data_dir: str,
    device: torch.device,
    num_samples_per_class: int = 2,
    save_dir: str = 'results/gradcam'
) -> None:
    """Generate Grad-CAM for samples from each class."""
    from . dataset_oasis import get_oasis_sample_images
    from .dataset import get_sample_images
    
    target_layer = get_target_layer(model, backbone_name)
    gradcam = GradCAM(model, target_layer, device)
    
    # Try OASIS first, then Kaggle
    try:
        samples = get_oasis_sample_images(data_dir, num_samples_per_class)
    except:
        samples = get_sample_images(data_dir, num_samples_per_class)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name, paths in samples.items():
        for i, path in enumerate(paths):
            try:
                gradcam.visualize(path, str(save_dir / f'{class_name}_{i}.png'))
                plt.close()
            except Exception as e:
                print(f"Error:  {e}")
    
    print(f"Grad-CAM saved to {save_dir}")