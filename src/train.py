"""
Training module for Alzheimer's MRI classification.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils. data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model. to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim. AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        
        self.history:  Dict[str, list] = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        }
        self.best_val_acc = 0.0
        self.best_model_state:  Optional[Dict] = None
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self. criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return running_loss / total, 100. * correct / total
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(self.val_loader, desc='Validating'):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss / total, 100. * correct / total
    
    def train(
        self,
        num_epochs: int = 30,
        save_dir: str = 'checkpoints',
        early_stopping_patience: int = 7
    ) -> Dict: 
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:  {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self. best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_dir / 'best_model.pth')
                print(f"✓ New best model saved!  Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    class_counts = torch.zeros(4)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    class_counts = torch.clamp(class_counts, min=1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    return weights