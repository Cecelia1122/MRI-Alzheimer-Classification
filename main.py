"""
Unified main script for Brain MRI Alzheimer's Classification. 

Supports:
    Datasets (Processed):
        - kaggle_2d:        Kaggle preprocessed 2D images (JPG)
        - kaggle_nifti_2d:  Kaggle NIfTI processed to 2D slices
        - kaggle_nifti_3d: Kaggle NIfTI processed to 3D volumes
        - oasis_2d:        OASIS processed to 2D slices
        - oasis_3d:        OASIS processed to 3D volumes
    
    Datasets (Raw - for preprocessing):
        - Kaggle:           Original Kaggle JPG images
        - kaggle_nifti_raw: Raw NIfTI files
        - oasis:            Raw OASIS data (disc1-disc12)

Usage:
    # ===================== PREPROCESSING =====================
    # OASIS -> 2D slices
    python main.py --mode preprocess --dataset oasis --dim 2d
    
    # OASIS -> 3D volumes
    python main.py --mode preprocess --dataset oasis --dim 3d
    
    # Kaggle NIfTI -> 2D slices
    python main.py --mode preprocess --dataset kaggle_nifti --dim 2d
    
    # Kaggle NIfTI -> 3D volumes
    python main.py --mode preprocess --dataset kaggle_nifti --dim 3d
    
    # Kaggle Original JPG -> Split into train/test
    python main.py --mode preprocess --dataset kaggle_orig

    # ===================== TRAINING =====================
    # Train on Kaggle 2D (preprocessed)
    python main.py --mode train --dataset kaggle_2d
    
    # Train on OASIS 2D
    python main.py --mode train --dataset oasis_2d
    
    # Train on OASIS 3D
    python main.py --mode train --dataset oasis_3d --batch_size 4 --backbone simple
    
    # Train on Kaggle NIfTI 2D
    python main.py --mode train --dataset kaggle_nifti_2d
    
    # Train on Kaggle NIfTI 3D
    python main.py --mode train --dataset kaggle_nifti_3d --batch_size 4 --backbone simple

    # ===================== EVALUATION =====================
    python main.py --mode evaluate --dataset oasis_2d
    
    # ===================== GRAD-CAM (2D only) =====================
    python main. py --mode gradcam --dataset oasis_2d
    
    # ===================== FULL PIPELINE =====================
    python main.py --mode full --dataset oasis_2d
"""

import argparse
from pathlib import Path
import shutil
import torch

# Preprocessing functions
from src.preprocessing import (
    prepare_oasis_2d,
    prepare_oasis_3d,
    prepare_kaggle_nifti_2d,
    prepare_kaggle_nifti_3d,
    split_original_kaggle_images
)

# Dataset loaders
from src.dataset import get_data_loaders
from src.dataset_oasis import get_oasis_data_loaders
from src.dataset_3d import get_3d_data_loaders

# Models
from src.model import get_model
from src. model_3d import get_3d_model

# Training and evaluation
from src.train import Trainer, compute_class_weights
from src. evaluate import Evaluator
from src.gradcam import visualize_samples_with_gradcam


# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_CONFIG = {
    # Processed datasets (ready for training)
    'kaggle_2d': {
        'path': './data/kaggle_2d',
        'dim': '2d',
        'loader': 'kaggle',
        'description': 'Kaggle preprocessed 2D images',
    },
    'kaggle_nifti_2d': {
        'path': './data/kaggle_nifti_2d',
        'dim': '2d',
        'loader': 'oasis',  # Same format as OASIS 2D
        'description': 'Kaggle NIfTI -> 2D slices',
    },
    'kaggle_nifti_3d': {
        'path': './data/kaggle_nifti_3d',
        'dim': '3d',
        'loader': '3d',
        'description':  'Kaggle NIfTI -> 3D volumes',
    },
    'oasis_2d': {
        'path': './data/oasis_2d',
        'dim': '2d',
        'loader':  'oasis',
        'description': 'OASIS -> 2D slices',
    },
    'oasis_3d': {
        'path':  './data/oasis_3d',
        'dim': '3d',
        'loader': '3d',
        'description':  'OASIS -> 3D volumes',
    },
    
    # Raw datasets (for preprocessing)
    'oasis': {
        'raw_path': './data/oasis',
        'output_2d':  './data/oasis_2d',
        'output_3d': './data/oasis_3d',
        'description': 'Raw OASIS data (disc1-disc12)',
    },
    'kaggle_nifti': {
        'raw_path': './data/kaggle_nifti_raw',
        'output_2d': './data/kaggle_nifti_2d',
        'output_3d': './data/kaggle_nifti_3d',
        'description': 'Raw Kaggle NIfTI files',
    },
    'kaggle_orig': {
        'raw_path': './data/Kaggle',
        'output_2d':  './data/kaggle_2d',
        'description': 'Original Kaggle JPG images',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Brain MRI Alzheimer\'s Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples: 
  # Preprocess OASIS to 2D slices
  python main.py --mode preprocess --dataset oasis --dim 2d
  
  # Train on OASIS 2D
  python main.py --mode train --dataset oasis_2d --epochs 30
  
  # Train on 3D volumes
  python main. py --mode train --dataset oasis_3d --batch_size 4 --backbone simple
  
  # Full pipeline
  python main.py --mode full --dataset kaggle_2d
        """
    )
    
    # Mode
    parser.add_argument(
        '--mode', type=str,
        choices=['preprocess', 'train', 'evaluate', 'gradcam', 'full'],
        default='train',
        help='Mode to run'
    )
    
    # Dataset selection
    parser.add_argument(
        '--dataset', type=str,
        choices=list(DATASET_CONFIG.keys()),
        default='oasis_2d',
        help='Dataset to use'
    )
    
    # Dimension (for preprocessing)
    parser.add_argument(
        '--dim', type=str,
        choices=['2d', '3d'],
        default='2d',
        help='Dimension for preprocessing (2d slices or 3d volumes)'
    )
    
    # Model
    parser.add_argument(
        '--backbone', type=str,
        default=None,
        help='Model backbone.  2D:  resnet18/resnet50/efficientnet_b0/densenet121.  3D: simple/resnet/resnet_small'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default=None,
        help='Path to checkpoint for evaluation/gradcam'
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: 32 for 2D, 4 for 3D)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_slices', type=int, default=20, help='Slices per subject (preprocessing)')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    
    args = parser.parse_args()
    
    # Set default backbone based on dimension
    if args.backbone is None:
        config = DATASET_CONFIG. get(args.dataset, {})
        dim = config.get('dim', args.dim)
        args.backbone = 'resnet50' if dim == '2d' else 'simple'
    
    # Set default batch size based on dimension
    if args.batch_size is None:
        config = DATASET_CONFIG.get(args.dataset, {})
        dim = config.get('dim', args.dim)
        args.batch_size = 64 if dim == '2d' else 4
    
    return args


def get_data_loader_for_dataset(dataset_name:  str, batch_size: int, num_workers: int = 4):
    """Get appropriate data loader based on dataset configuration."""
    config = DATASET_CONFIG[dataset_name]
    data_path = config['path']
    loader_type = config['loader']
    
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Please run preprocessing first or check your data directory."
        )
    
    if loader_type == 'kaggle':
        return get_data_loaders(data_path, batch_size, num_workers)
    elif loader_type == 'oasis': 
        return get_oasis_data_loaders(data_path, batch_size, num_workers)
    elif loader_type == '3d':
        return get_3d_data_loaders(data_path, batch_size, num_workers)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")


def preprocess(args):
    """Run preprocessing based on dataset and dimension."""
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)
    
    dataset = args.dataset
    dim = args. dim
    
    if dataset == 'oasis': 
        config = DATASET_CONFIG['oasis']
        raw_path = config['raw_path']
        
        if dim == '2d':
            output_path = config['output_2d']
            print(f"Processing OASIS -> 2D slices")
            print(f"  Input:   {raw_path}")
            print(f"  Output: {output_path}")
            prepare_oasis_2d(raw_path, output_path, num_slices=args.num_slices)
        else: 
            output_path = config['output_3d']
            print(f"Processing OASIS -> 3D volumes")
            print(f"  Input:  {raw_path}")
            print(f"  Output: {output_path}")
            prepare_oasis_3d(raw_path, output_path)
    
    elif dataset == 'kaggle_nifti': 
        config = DATASET_CONFIG['kaggle_nifti']
        raw_path = config['raw_path']
        
        if dim == '2d': 
            output_path = config['output_2d']
            print(f"Processing Kaggle NIfTI -> 2D slices")
            print(f"  Input:  {raw_path}")
            print(f"  Output: {output_path}")
            prepare_kaggle_nifti_2d(raw_path, output_path, num_slices=args.num_slices)
        else:
            output_path = config['output_3d']
            print(f"Processing Kaggle NIfTI -> 3D volumes")
            print(f"  Input:  {raw_path}")
            print(f"  Output: {output_path}")
            prepare_kaggle_nifti_3d(raw_path, output_path)
    
    elif dataset == 'kaggle_orig':
        config = DATASET_CONFIG['kaggle_orig']
        raw_path = config['raw_path']
        output_path = config['output_2d']
        print(f"Splitting Kaggle JPG images -> train/test")
        print(f"  Input:  {raw_path}")
        print(f"  Output: {output_path}")
        split_original_kaggle_images(raw_path, output_path)
    
    else:
        print(f"Dataset '{dataset}' is already processed.  No preprocessing needed.")
        print(f"Use one of: oasis, kaggle_nifti, kaggle_orig")


def train(args):
    """Train model on specified dataset."""
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    if args.dim == '3d':
    # 3D 算子在 MPS 上不完整，强制使用 CPU 以确保稳定
        device = torch.device("cpu")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")    
    print(f"Device: {device}")
    
    dataset = args.dataset
    config = DATASET_CONFIG[dataset]
    dim = config['dim']
    
    print(f"Dataset: {dataset} ({config['description']})")
    print(f"Dimension: {dim}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loader_for_dataset(dataset, args.batch_size)
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader)
    print(f"Class weights: {class_weights}")
    
    # Create model
    if dim == '2d':
        model = get_model(
            backbone=args.backbone,
            num_classes=4,
            pretrained=True,
            device=device,
            model_dim='2d'
        )
    else:
        model = get_3d_model(
            model_type=args.backbone,
            num_classes=4,
            device=device
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        class_weights=class_weights
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        early_stopping_patience=7
    )
    
    # Save with dataset-specific name
    checkpoint_name = f'best_model_{dataset}.pth'
    src = Path(args.save_dir) / 'best_model.pth'
    dst = Path(args.save_dir) / checkpoint_name
    if src.exists():
        shutil.copy(src, dst)
        print(f"✓ Model saved to:  {dst}")
    
    # Plot training history
    results_dir = Path(args.results_dir) / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(model, test_loader, device)
    evaluator.plot_training_history(
        history,
        str(results_dir / f'training_history_{dataset}.png')
    )
    print(f"✓ Training history saved to: {results_dir}/training_history_{dataset}.png")
    
    return model, history


def evaluate(args, model=None):
    """Evaluate model on specified dataset."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    if args.dim == '3d':
    # 3D 算子在 MPS 上不完整，强制使用 CPU 以确保稳定
        device = torch.device("cpu")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu") 
    print(f"Device: {device}")
 
    
    dataset = args.dataset
    config = DATASET_CONFIG[dataset]
    dim = config['dim']
    
    # Load model if not provided
    if model is None: 
        if dim == '2d':
            model = get_model(
                backbone=args.backbone,
                num_classes=4,
                pretrained=False,
                device=device,
                model_dim='2d'
            )
        else:
            model = get_3d_model(
                model_type=args.backbone,
                num_classes=4,
                device=device
            )
        
        # Find checkpoint
        checkpoint_paths = [
            Path(args.checkpoint) if args.checkpoint else None,
            Path(args.save_dir) / f'best_model_{dataset}.pth',
            Path(args.save_dir) / 'best_model.pth',
        ]
        
        loaded = False
        for cp in checkpoint_paths:
            if cp and cp.exists():
                checkpoint = torch.load(cp, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded checkpoint:  {cp}")
                if 'val_acc' in checkpoint:
                    print(f"  Validation accuracy: {checkpoint['val_acc']:.2f}%")
                loaded = True
                break
        
        if not loaded:
            print("✗ No checkpoint found!")
            print("  Searched:")
            for cp in checkpoint_paths:
                if cp:
                    print(f"    - {cp}")
            return None
    
    # Get test loader
    _, test_loader = get_data_loader_for_dataset(dataset, args.batch_size)
    
    # Evaluate
    results_dir = Path(args.results_dir) / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(model, test_loader, device)
    results = evaluator.full_evaluation(str(results_dir))
    
    print(f"\n✓ Results saved to: {results_dir}/")
    
    return results


def gradcam(args, model=None):
    """Generate Grad-CAM visualizations (2D only)."""
    print("\n" + "=" * 60)
    print("GRAD-CAM VISUALIZATION")
    print("=" * 60)
    
    dataset = args.dataset
    config = DATASET_CONFIG[dataset]
    dim = config['dim']
    
    if dim == '3d':
        print("⚠ Grad-CAM is only supported for 2D models")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if not provided
    if model is None:
        model = get_model(
            backbone=args.backbone,
            num_classes=4,
            pretrained=False,
            device=device,
            model_dim='2d'
        )
        
        checkpoint_paths = [
            Path(args.checkpoint) if args.checkpoint else None,
            Path(args.save_dir) / f'best_model_{dataset}.pth',
            Path(args.save_dir) / 'best_model.pth',
        ]
        
        for cp in checkpoint_paths:
            if cp and cp.exists():
                checkpoint = torch.load(cp, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded checkpoint: {cp}")
                break
    
    # Generate visualizations
    gradcam_dir = Path(args.results_dir) / 'gradcam'
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = config['path']
    
    try:
        visualize_samples_with_gradcam(
            model=model,
            backbone_name=args.backbone,
            data_dir=data_path,
            device=device,
            num_samples_per_class=3,
            save_dir=str(gradcam_dir)
        )
        print(f"✓ Grad-CAM visualizations saved to: {gradcam_dir}/")
    except Exception as e:
        print(f"⚠ Grad-CAM failed: {e}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print(" BRAIN MRI ALZHEIMER'S CLASSIFICATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    
    # Create directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    model = None
    
    # ==================== PREPROCESS ====================
    if args. mode == 'preprocess': 
        preprocess(args)
        return
    
    # ==================== TRAIN ====================
    if args.mode in ['train', 'full']: 
        model, _ = train(args)
    
    # ==================== EVALUATE ====================
    if args.mode in ['evaluate', 'full']:
        evaluate(args, model)
    
    # ==================== GRAD-CAM ====================
    if args.mode in ['gradcam', 'full']:
        gradcam(args, model)
    
    # ==================== COMPLETE ====================
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Checkpoints:  {args.save_dir}/")
    print(f"Results: {args.results_dir}/")


if __name__ == '__main__':
    main()