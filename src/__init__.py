"""
Brain MRI Alzheimer's Classification Package

A comprehensive deep learning framework for classifying Alzheimer's disease stages
from brain MRI scans. 

Supports:
    Datasets: 
        - Kaggle Alzheimer's Dataset (preprocessed 2D images)
        - Kaggle NIfTI Dataset (raw 3D volumes)
        - OASIS-1 Cross-sectional Dataset (raw 3D volumes)
    
    Modes:
        - 2D: Extract slices from 3D volumes, use pretrained CNNs
        - 3D: Use full 3D volumes with 3D CNNs

    Models:
        - 2D: ResNet18, ResNet50, EfficientNet-B0, DenseNet121
        - 3D: Simple3DCNN, ResNet3D

Classes:
    0:  NonDemented (CDR = 0)
    1: VeryMildDemented (CDR = 0.5)
    2: MildDemented (CDR = 1)
    3: ModerateDemented (CDR >= 2)

Usage:
    # 2D Pipeline
    from src import get_model, get_data_loaders, Trainer, Evaluator
    
    model = get_model(backbone='resnet50', model_dim='2d')
    train_loader, test_loader = get_data_loaders('./data/kaggle')
    trainer = Trainer(model, train_loader, test_loader, device)
    trainer.train(num_epochs=30)
    
    # 3D Pipeline
    from src import get_3d_model, get_3d_data_loaders
    
    model = get_3d_model(model_type='resnet')
    train_loader, test_loader = get_3d_data_loaders('./data/oasis_3d')
"""

# =============================================================================
# Version and Metadata
# =============================================================================
__version__ = '2.0.0'
__author__ = 'Xue Li'
__description__ = 'Brain MRI Alzheimer\'s Classification with 2D and 3D Deep Learning'

# Class names for reference
CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
NUM_CLASSES = 4

# =============================================================================
# Preprocessing Module
# =============================================================================
from . preprocessing import (
    # Image loading
    load_nifti,
    load_analyze,
    load_mri,
    
    # Volume processing
    normalize_volume,
    resize_volume,
    extract_slices,
    resize_slice,
    extract_center_slice,
    
    # Label conversion
    get_cdr_label,
    
    # OASIS utilities
    find_oasis_csv,
    find_oasis_image,
    find_all_oasis_subjects,
    
    # Dataset preparation - 2D
    prepare_oasis_2d,
    prepare_kaggle_nifti_2d,
    
    # Dataset preparation - 3D
    prepare_oasis_3d,
    prepare_kaggle_nifti_3d,
)

# =============================================================================
# 2D Dataset Module (Kaggle preprocessed images)
# =============================================================================
from .dataset import (
    AlzheimerMRIDataset,
    get_data_loaders,
    get_sample_images,
)

# =============================================================================
# 2D Dataset Module (OASIS processed slices)
# =============================================================================
from .dataset_oasis import (
    OASISSliceDataset,
    get_oasis_data_loaders,
    get_oasis_sample_images,
)

# =============================================================================
# 3D Dataset Module
# =============================================================================
from .dataset_3d import (
    MRI3DDataset,
    Transform3D,
    get_3d_data_loaders,
)

# =============================================================================
# 2D Model Module
# =============================================================================
from .model import (
    AlzheimerClassifier2D,
    get_model,
)

# =============================================================================
# 3D Model Module
# =============================================================================
from .model_3d import (
    Simple3DCNN,
    ResNet3D,
    ResBlock3D,
    ConvBlock3D,
    get_3d_model,
)

# =============================================================================
# Training Module
# =============================================================================
from .train import (
    Trainer,
    compute_class_weights,
)

# =============================================================================
# Evaluation Module
# =============================================================================
from .evaluate import (
    Evaluator,
)

# =============================================================================
# Grad-CAM Module (2D only)
# =============================================================================
from .gradcam import (
    GradCAM,
    get_target_layer,
    visualize_samples_with_gradcam,
)

# =============================================================================
# Convenience Functions
# =============================================================================

def get_dataset_info(dataset_type:  str) -> dict:
    """
    Get information about supported datasets.
    
    Args:
        dataset_type: 'kaggle', 'kaggle_nifti', or 'oasis'
    
    Returns:
        Dictionary with dataset information
    """
    datasets = {
        'kaggle':  {
            'name': 'Kaggle Alzheimer\'s Dataset',
            'format': 'Preprocessed 2D images (JPG/PNG)',
            'source': 'https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images',
            'default_path': './data/kaggle',
            'supports_2d': True,
            'supports_3d': False,
            'requires_preprocessing': False,
        },
        'kaggle_nifti': {
            'name': 'Kaggle Alzheimer\'s NIfTI Dataset',
            'format': 'NIfTI 3D volumes (. nii, .nii.gz)',
            'source': 'Kaggle',
            'default_path': './data/kaggle_nifti_raw',
            'supports_2d': True,
            'supports_3d': True,
            'requires_preprocessing': True,
        },
        'oasis': {
            'name': 'OASIS-1 Cross-sectional Dataset',
            'format': 'Analyze 7. 5 format (. img/. hdr)',
            'source': 'https://www.oasis-brains.org/',
            'default_path': './data/oasis',
            'supports_2d': True,
            'supports_3d': True,
            'requires_preprocessing':  True,
            'num_subjects': 416,
            'num_discs': 12,
        },
    }
    
    if dataset_type not in datasets: 
        raise ValueError(f"Unknown dataset:  {dataset_type}. Choose from {list(datasets.keys())}")
    
    return datasets[dataset_type]


def get_model_info(model_dim: str = '2d') -> dict:
    """
    Get information about supported models.
    
    Args:
        model_dim: '2d' or '3d'
    
    Returns:
        Dictionary with model information
    """
    if model_dim == '2d': 
        return {
            'supported_backbones': ['resnet18', 'resnet50', 'efficientnet_b0', 'densenet121'],
            'default_backbone': 'resnet50',
            'input_shape': (3, 224, 224),
            'pretrained_available': True,
            'recommended_batch_size': 32,
        }
    elif model_dim == '3d': 
        return {
            'supported_backbones': ['simple', 'resnet', 'resnet_small'],
            'default_backbone': 'simple',
            'input_shape': (1, 128, 128, 128),
            'pretrained_available': False,
            'recommended_batch_size': 4,
        }
    else: 
        raise ValueError(f"model_dim must be '2d' or '3d', got {model_dim}")


def create_model(
    model_dim: str = '2d',
    backbone: str = None,
    num_classes: int = 4,
    pretrained: bool = True,
    device=None
):
    """
    Convenience function to create a model.
    
    Args:
        model_dim: '2d' or '3d'
        backbone: Model backbone (None = use default)
        num_classes:  Number of output classes
        pretrained: Use pretrained weights (2D only)
        device: Torch device
    
    Returns:
        Initialized model
    
    Example:
        >>> model = create_model('2d', backbone='resnet50')
        >>> model = create_model('3d', backbone='simple')
    """
    info = get_model_info(model_dim)
    
    if backbone is None:
        backbone = info['default_backbone']
    
    if backbone not in info['supported_backbones']:
        raise ValueError(
            f"Backbone '{backbone}' not supported for {model_dim}. "
            f"Choose from {info['supported_backbones']}"
        )
    
    if model_dim == '2d': 
        return get_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device,
            model_dim='2d'
        )
    else:
        return get_3d_model(
            model_type=backbone,
            num_classes=num_classes,
            device=device
        )


def create_data_loaders(
    data_dir: str,
    dataset_type: str = 'kaggle',
    model_dim: str = '2d',
    batch_size: int = None,
    num_workers: int = 4
):
    """
    Convenience function to create data loaders. 
    
    Args:
        data_dir: Path to processed data
        dataset_type: 'kaggle', 'kaggle_nifti', or 'oasis'
        model_dim: '2d' or '3d'
        batch_size: Batch size (None = use default)
        num_workers: Number of data loading workers
    
    Returns: 
        (train_loader, test_loader)
    
    Example:
        >>> train_loader, test_loader = create_data_loaders('./data/kaggle', 'kaggle', '2d')
        >>> train_loader, test_loader = create_data_loaders('./data/oasis_3d', 'oasis', '3d')
    """
    info = get_model_info(model_dim)
    
    if batch_size is None:
        batch_size = info['recommended_batch_size']
    
    if model_dim == '3d':
        return get_3d_data_loaders(data_dir, batch_size, num_workers)
    elif dataset_type == 'kaggle': 
        return get_data_loaders(data_dir, batch_size, num_workers)
    else:  # oasis or kaggle_nifti (processed to 2D)
        return get_oasis_data_loaders(data_dir, batch_size, num_workers)


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Metadata
    '__version__',
    '__author__',
    'CLASS_NAMES',
    'NUM_CLASSES',
    
    # Preprocessing
    'load_nifti',
    'load_analyze',
    'load_mri',
    'normalize_volume',
    'resize_volume',
    'extract_slices',
    'resize_slice',
    'extract_center_slice',
    'get_cdr_label',
    'find_oasis_csv',
    'find_oasis_image',
    'find_all_oasis_subjects',
    'prepare_oasis_2d',
    'prepare_oasis_3d',
    'prepare_kaggle_nifti_2d',
    'prepare_kaggle_nifti_3d',
    
    # 2D Datasets
    'AlzheimerMRIDataset',
    'get_data_loaders',
    'get_sample_images',
    'OASISSliceDataset',
    'get_oasis_data_loaders',
    'get_oasis_sample_images',
    
    # 3D Datasets
    'MRI3DDataset',
    'Transform3D',
    'get_3d_data_loaders',
    
    # 2D Models
    'AlzheimerClassifier2D',
    'get_model',
    
    # 3D Models
    'Simple3DCNN',
    'ResNet3D',
    'ResBlock3D',
    'ConvBlock3D',
    'get_3d_model',
    
    # Training
    'Trainer',
    'compute_class_weights',
    
    # Evaluation
    'Evaluator',
    
    # Grad-CAM
    'GradCAM',
    'get_target_layer',
    'visualize_samples_with_gradcam',
    
    # Convenience functions
    'get_dataset_info',
    'get_model_info',
    'create_model',
    'create_data_loaders',
]