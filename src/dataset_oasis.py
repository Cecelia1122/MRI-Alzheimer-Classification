"""
Dataset module for OASIS processed slices.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class OASISSliceDataset(Dataset):
    """Dataset for processed OASIS slices."""
    
    CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Directory not found: {self.split_dir}\n"
                "Run preprocessing first:  python main.py --mode preprocess --dataset oasis"
            )
        
        self.transform = transform if transform else self._get_default_transform(split)
        
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        
        for class_name in self.CLASS_NAMES:
            class_dir = self.split_dir / class_name
            if class_dir.exists():
                for ext in ['*.png', '*.jpg']: 
                    for img_path in class_dir.glob(ext):
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.split_dir}")
        
        print(f"[OASIS] Loaded {len(self.samples)} slices for {split}")
    
    def _get_default_transform(self, split: str) -> transforms.Compose:
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]: 
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        distribution = {name: 0 for name in self.CLASS_NAMES}
        for _, label in self.samples:
            distribution[self.CLASS_NAMES[label]] += 1
        return distribution


def get_oasis_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers:  int = 4
) -> Tuple[DataLoader, DataLoader]: 
    """Create OASIS data loaders."""
    train_dataset = OASISSliceDataset(data_dir, split='train')
    test_dataset = OASISSliceDataset(data_dir, split='test')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_oasis_sample_images(data_dir: str, num_samples:  int = 4) -> Dict[str, List[str]]: 
    """Get sample images for visualization."""
    dataset = OASISSliceDataset(data_dir, split='train')
    samples: Dict[str, List[str]] = {name: [] for name in dataset.CLASS_NAMES}
    
    for img_path, label in dataset.samples:
        class_name = dataset.CLASS_NAMES[label]
        if len(samples[class_name]) < num_samples:
            samples[class_name].append(img_path)
    
    return samples