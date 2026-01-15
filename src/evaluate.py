"""
Evaluation module. 
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from tqdm import tqdm


class Evaluator:
    """Model evaluator with visualization."""
    
    CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, device:  torch.device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.predictions = None
        self.probabilities = None
        self.true_labels = None
    
    @torch.no_grad()
    def run_inference(self) -> None:
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        for images, labels in tqdm(self.test_loader, desc='Evaluating'):
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels. extend(labels.numpy())
        
        self.predictions = np.array(all_preds)
        self.probabilities = np.array(all_probs)
        self.true_labels = np.array(all_labels)
    
    def get_accuracy(self) -> float:
        if self.predictions is None:
            self.run_inference()
        return float((self.predictions == self.true_labels).mean() * 100)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        if self.predictions is None:
            self.run_inference()
        
        cm = confusion_matrix(self.true_labels, self. predictions)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self. CLASS_NAMES, yticklabels=self.CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_roc_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        if self. probabilities is None: 
            self.run_inference()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (name, color) in enumerate(zip(self. CLASS_NAMES, colors)):
            y_true = (self.true_labels == i).astype(int)
            y_score = self.probabilities[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1]. plot(epochs, history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def full_evaluation(self, save_dir: str = 'results/figures') -> Dict:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_inference()
        accuracy = self.get_accuracy()
        report = classification_report(self.true_labels, self.predictions,labels=[0, 1, 2, 3],  # 强制包含所有类别索引
                                    target_names=self.CLASS_NAMES, digits=4)

        
        print("\n" + "=" * 60)
        print(f"RESULTS - Accuracy: {accuracy:.2f}%")
        print("=" * 60)
        print(report)
        
        self.plot_confusion_matrix(str(save_dir / 'confusion_matrix.png'))
        self.plot_roc_curves(str(save_dir / 'roc_curves.png'))
        plt.close('all')
        
        return {'accuracy': accuracy, 'report':  report}