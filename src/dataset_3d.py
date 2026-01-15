"""
3D Volume Dataset for brain MRI classification.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader


class MRI3DDataset(Dataset):
    """Dataset for 3D MRI volumes stored as . npy files."""
    
    CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform:  Optional[callable] = None,
        target_shape: Tuple[int, int, int] = (128, 128, 128)
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.transform = transform
        self.target_shape = target_shape
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.split_dir}")
        
        self.samples:  List[Tuple[str, int]] = []
        self.class_to_idx = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        
        for class_name in self.CLASS_NAMES:
            class_dir = self.split_dir / class_name
            if class_dir.exists():
                for npy_path in class_dir.glob('*.npy'):
                    self.samples.append((str(npy_path), self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No . npy files found in {self. split_dir}")
        
        print(f"[3D] Loaded {len(self.samples)} volumes for {split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]: 
        npy_path, label = self.samples[idx]
        
        # Load volume
        volume = np.load(npy_path).astype(np.float32)
        
        # Apply transforms if any
        if self.transform:
            volume = self.transform(volume)
        
        # Add channel dimension:  (D, H, W) -> (1, D, H, W)
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).unsqueeze(0)
        elif len(volume.shape) == 3:
            volume = volume.unsqueeze(0)
        
        return volume, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        distribution = {name: 0 for name in self.CLASS_NAMES}
        for _, label in self.samples:
            distribution[self.CLASS_NAMES[label]] += 1
        return distribution


class Transform3D:
    """Basic 3D augmentation transforms."""
    
    def __init__(self, train:  bool = True):
        self.train = train
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if self.train:
            # Random flip along each axis
            if np.random.random() > 0.5:
                volume = np.flip(volume, axis=0).copy()
            if np.random.random() > 0.5:
                volume = np.flip(volume, axis=1).copy()
            if np.random. random() > 0.5:
                volume = np.flip(volume, axis=2).copy()
            
            # Random intensity shift
            if np.random. random() > 0.5:
                shift = np.random.uniform(-0.1, 0.1)
                volume = np.clip(volume + shift, 0, 1)
            
            # Random intensity scale
            if np.random. random() > 0.5:
                scale = np.random. uniform(0.9, 1.1)
                volume = np.clip(volume * scale, 0, 1)
        
        return volume


def get_3d_data_loaders(
    data_dir:  str,
    batch_size:  int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create 3D data loaders."""
    train_dataset = MRI3DDataset(
        data_dir, split='train',
        transform=Transform3D(train=True)
    )
    test_dataset = MRI3DDataset(
        data_dir, split='test',
        transform=Transform3D(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader