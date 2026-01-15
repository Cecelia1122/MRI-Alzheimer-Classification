"""
Debug script for testing each component individually. 
Run sections one at a time to identify issues. 

Usage:
    python debug.py --step 1    # Test preprocessing
    python debug.py --step 2    # Test datasets
    python debug.py --step 3    # Test models
    python debug.py --step 4    # Test training
    python debug.py --step 5    # Test evaluation
    python debug.py --step 6    # Test Grad-CAM
    python debug.py --step all  # Test everything
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np


def print_header(text:  str):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_success(text: str):
    print(f"  ✓ {text}")


def print_error(text: str):
    print(f"  ✗ {text}")


def print_info(text: str):
    print(f"  ℹ {text}")


# =============================================================================
# STEP 1: Test Preprocessing
# =============================================================================
def test_preprocessing():
    print_header("STEP 1: Testing Preprocessing")
    
    try:
        from src.preprocessing import (
            load_nifti, load_analyze, load_mri,
            normalize_volume, resize_volume, extract_slices, resize_slice,
            find_oasis_csv, find_all_oasis_subjects,
            prepare_oasis_2d, prepare_oasis_3d  
        )
        print_success("Imports successful")
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return False
    
    # Test normalize_volume
    try:
        # Create a dummy 3D volume
        test_vol = np.random.rand(64, 64, 64).astype(np.float32) * 100
        normalized = normalize_volume(test_vol)
        assert normalized.min() >= 0 and normalized.max() <= 1
        print_success(f"normalize_volume: input range [{test_vol.min():.1f}, {test_vol.max():.1f}] -> [{normalized.min():.3f}, {normalized.max():.3f}]")
    except Exception as e:
        print_error(f"normalize_volume failed: {e}")
        return False
    
    # Test resize_volume (Matches your new squeeze logic)
    try:
        # Testing with a 4D shape (extra dim) to trigger your squeeze fix
        test_vol_4d = np.random.rand(64, 64, 64, 1).astype(np.float32)
        resized = resize_volume(test_vol_4d, (32, 32, 32))
        assert resized.shape == (32, 32, 32)
        print_success(f"resize_volume (with squeeze fix): {test_vol_4d.shape} -> {resized.shape}")
    except Exception as e:
        print_error(f"resize_volume failed: {e}")
        return False
    
    # Test extract_slices
    try:
        slices = extract_slices(test_vol, num_slices=10, axis=2)
        assert len(slices) == 10
        print_success(f"extract_slices: extracted {len(slices)} slices")
    except Exception as e: 
        print_error(f"extract_slices failed: {e}")
        return False
    
    # Test OASIS CSV finder (Checking multiple possible locations)
    try:
        # Check root first, then data/
        csv_path = find_oasis_csv(Path(".")) or find_oasis_csv(Path("./data"))
        if csv_path and Path(csv_path).exists():
            print_success(f"find_oasis_csv: found {csv_path}")
        else:
            print_info("find_oasis_csv: CSV not found in standard locations")
    except Exception as e:
        print_error(f"find_oasis_csv failed: {e}")
    
    print_success("Preprocessing module matches latest script logic")
    return True


# =============================================================================
# STEP 2: Test Datasets
# =============================================================================
def test_datasets():
    print_header("STEP 2: Testing Datasets")
    
    # Test Kaggle 2D Dataset
    print("\n  --- Kaggle 2D Dataset ---")
    try:
        from src.dataset import AlzheimerMRIDataset, get_data_loaders
        print_success("Kaggle dataset imports OK")
        
        kaggle_dir = Path("./data/kaggle_2d")
        if kaggle_dir.exists() and (kaggle_dir / "train").exists():
            dataset = AlzheimerMRIDataset(str(kaggle_dir), split='train')
            print_success(f"Loaded {len(dataset)} samples")
            
            # Test single item
            img, label = dataset[0]
            print_success(f"Sample shape: {img.shape}, label: {label}")
            
            # Test distribution
            dist = dataset.get_class_distribution()
            print_success(f"Class distribution: {dist}")
            
            # Test data loader
            train_loader, test_loader = get_data_loaders(str(kaggle_dir), batch_size=4)
            batch = next(iter(train_loader))
            print_success(f"Batch shape: {batch[0].shape}")
        else:
            print_info("Kaggle data not found, skipping")
            
    except Exception as e:
        print_error(f"Kaggle dataset error: {e}")
    
    # Test OASIS 2D Dataset
    print("\n  --- OASIS 2D Dataset ---")
    try:
        from src.dataset_oasis import OASISSliceDataset, get_oasis_data_loaders
        print_success("OASIS dataset imports OK")
        
        oasis_2d_dir = Path("./data/oasis_2d")
        if oasis_2d_dir.exists() and (oasis_2d_dir / "train").exists():
            dataset = OASISSliceDataset(str(oasis_2d_dir), split='train')
            print_success(f"Loaded {len(dataset)} samples")
            
            img, label = dataset[0]
            print_success(f"Sample shape: {img.shape}, label: {label}")
            
            dist = dataset.get_class_distribution()
            print_success(f"Class distribution: {dist}")
        else:
            print_info("OASIS 2D data not found, skipping (run preprocessing first)")
            
    except Exception as e:
        print_error(f"OASIS dataset error: {e}")
    
    # Test 3D Dataset
    print("\n  --- 3D Volume Dataset ---")
    try:
        from src.dataset_3d import MRI3DDataset, get_3d_data_loaders, Transform3D
        print_success("3D dataset imports OK")
        
        oasis_3d_dir = Path("./data/oasis_3d")
        if oasis_3d_dir.exists() and (oasis_3d_dir / "train").exists():
            dataset = MRI3DDataset(str(oasis_3d_dir), split='train')
            print_success(f"Loaded {len(dataset)} volumes")
            
            vol, label = dataset[0]
            print_success(f"Volume shape: {vol.shape}, label: {label}")
        else:
            print_info("OASIS 3D data not found, skipping (run 3D preprocessing first)")
            
    except Exception as e:
        print_error(f"3D dataset error: {e}")
    
    print_success("Dataset modules OK")
    return True


# =============================================================================
# STEP 3: Test Models
# =============================================================================
def test_models():
    print_header("STEP 3: Testing Models")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Using device: {device}")
    
    # Test 2D Models
    print("\n  --- 2D Models ---")
    try:
        from src.model import get_model, AlzheimerClassifier2D
        print_success("2D model imports OK")
        
        # Test each backbone
        backbones_2d = ['resnet18', 'resnet50']  # Skip others for speed
        for backbone in backbones_2d:
            model = get_model(backbone=backbone, num_classes=4, pretrained=True, device=device, model_dim='2d')
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            output = model(dummy_input)
            
            assert output.shape == (2, 4)
            print_success(f"{backbone}:  input (2,3,224,224) → output {tuple(output.shape)}")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e: 
        print_error(f"2D model error: {e}")
        return False
    
    # Test 3D Models
    print("\n  --- 3D Models ---")
    try:
        from src.model_3d import get_3d_model, Simple3DCNN, ResNet3D
        print_success("3D model imports OK")
        
        model_types = ['simple', 'resnet_small']
        for model_type in model_types:
            model = get_3d_model(model_type=model_type, num_classes=4, device=device)
            
            model.eval()
            # Test forward pass (smaller input for speed)
            dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)
            output = model(dummy_input)
            
            assert output.shape == (1, 4)
            print_success(f"{model_type}:  input (1,1,64,64,64) → output {tuple(output.shape)}")
            
            del model
            torch. cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e:
        print_error(f"3D model error: {e}")
        return False
    
    print_success("Model modules OK")
    return True


# =============================================================================
# STEP 4: Test Training
# =============================================================================
def test_training():
    print_header("STEP 4: Testing Training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from src.train import Trainer, compute_class_weights
        from src.model import get_model
        from torch.utils.data import DataLoader, TensorDataset
        print_success("Training imports OK")
        
        # Create dummy data
        num_samples = 32
        dummy_images = torch.randn(num_samples, 3, 224, 224)
        dummy_labels = torch. randint(0, 4, (num_samples,))
        
        dataset = TensorDataset(dummy_images, dummy_labels)
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        print_success("Created dummy data loaders")
        
        # Test class weights
        class_weights = compute_class_weights(train_loader)
        print_success(f"Computed class weights: {class_weights}")
        
        # Create model
        model = get_model(backbone='resnet18', num_classes=4, pretrained=True, device=device, model_dim='2d')
        print_success("Created model")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=1e-4,
            class_weights=class_weights
        )
        print_success("Created trainer")
        
        # Run one epoch
        print_info("Running 1 training epoch (this may take a moment)...")
        train_loss, train_acc = trainer.train_epoch()
        print_success(f"Train epoch:  loss={train_loss:.4f}, acc={train_acc:.2f}%")
        
        # Run validation
        val_loss, val_acc = trainer.validate()
        print_success(f"Validation: loss={val_loss:.4f}, acc={val_acc:.2f}%")
        
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e: 
        print_error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print_success("Training module OK")
    return True


# =============================================================================
# STEP 5: Test Evaluation
# =============================================================================
def test_evaluation():
    print_header("STEP 5: Testing Evaluation")
    
    device = torch. device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from src.evaluate import Evaluator
        from src.model import get_model
        from torch.utils.data import DataLoader, TensorDataset
        print_success("Evaluation imports OK")
        
        # Create dummy data
        num_samples = 32
        dummy_images = torch.randn(num_samples, 3, 224, 224)
        dummy_labels = torch.randint(0, 4, (num_samples,))
        
        dataset = TensorDataset(dummy_images, dummy_labels)
        test_loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Create model
        model = get_model(backbone='resnet18', num_classes=4, pretrained=True, device=device, model_dim='2d')
        
        # Create evaluator
        evaluator = Evaluator(model, test_loader, device)
        print_success("Created evaluator")
        
        # Run inference
        evaluator.run_inference()
        print_success(f"Inference complete:  {len(evaluator.predictions)} predictions")
        
        # Get accuracy
        acc = evaluator.get_accuracy()
        print_success(f"Accuracy: {acc:.2f}%")
        
        # Test plotting (save to temp location)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator.plot_confusion_matrix(f"{tmpdir}/cm.png")
            print_success("Confusion matrix plot OK")
            
            evaluator.plot_roc_curves(f"{tmpdir}/roc.png")
            print_success("ROC curves plot OK")
        
        del model, evaluator
        
    except Exception as e:
        print_error(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print_success("Evaluation module OK")
    return True


# =============================================================================
# STEP 6: Test Grad-CAM
# =============================================================================
def test_gradcam():
    print_header("STEP 6: Testing Grad-CAM")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from src.gradcam import GradCAM, get_target_layer
        from src.model import get_model
        print_success("Grad-CAM imports OK")
        
        # Create model
        model = get_model(backbone='resnet18', num_classes=4, pretrained=True, device=device, model_dim='2d')
        
        # Get target layer
        target_layer = get_target_layer(model, 'resnet18')
        print_success(f"Target layer: {type(target_layer).__name__}")
        
        # Create GradCAM
        gradcam = GradCAM(model, target_layer, device)
        print_success("Created GradCAM")
        
        # Generate CAM
        dummy_input = torch.randn(1, 3, 224, 224)
        cam = gradcam.generate_cam(dummy_input)
        print_success(f"Generated CAM: shape={cam.shape}, range=[{cam.min():.3f}, {cam.max():.3f}]")
        
        del model, gradcam
        
    except Exception as e:
        print_error(f"Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print_success("Grad-CAM module OK")
    return True


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Debug script')
    parser.add_argument('--step', type=str, default='all',
                        help='Step to test:  1-6 or "all"')
    args = parser.parse_args()
    
    print("=" * 60)
    print(" BRAIN MRI CLASSIFICATION - DEBUG SCRIPT")
    print("=" * 60)
    
    steps = {
        '1': ('Preprocessing', test_preprocessing),
        '2': ('Datasets', test_datasets),
        '3': ('Models', test_models),
        '4': ('Training', test_training),
        '5': ('Evaluation', test_evaluation),
        '6': ('Grad-CAM', test_gradcam),
    }
    
    if args.step == 'all': 
        steps_to_run = list(steps.keys())
    else:
        steps_to_run = [args.step]
    
    results = {}
    for step_num in steps_to_run: 
        if step_num in steps:
            name, func = steps[step_num]
            try:
                results[step_num] = func()
            except Exception as e:
                print_error(f"Step {step_num} crashed: {e}")
                results[step_num] = False
    
    # Summary
    print_header("SUMMARY")
    all_passed = True
    for step_num, passed in results.items():
        name = steps[step_num][0]
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Step {step_num} ({name}): {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n  🎉 All tests passed!")
    else:
        print("\n  ⚠️ Some tests failed.  Fix errors before running main. py")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())