"""
Preprocessing module for brain MRI data. 
Supports:  OASIS (Analyze format), Kaggle NIfTI, 2D slices, 3D volumes.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try: 
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError: 
    SCIPY_AVAILABLE = False


def check_dependencies():
    if not NIBABEL_AVAILABLE: 
        raise ImportError("nibabel required:  pip install nibabel")
    if not SCIPY_AVAILABLE: 
        raise ImportError("scipy required:  pip install scipy")


# =============================================================================
# Image Loading Functions
# =============================================================================

def load_nifti(path: str) -> np.ndarray:
    """Load NIfTI image (.nii or .nii.gz)."""
    check_dependencies()
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)


def load_analyze(path: str) -> np.ndarray:
    """Load Analyze 7.5 image (.img/.hdr)."""
    check_dependencies()
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)


def load_mri(path: str) -> np.ndarray:
    """Load MRI image (auto-detect format)."""
    path = str(path).strip()
    if path.endswith('.nii') or path.endswith('.nii.gz'):
        return load_nifti(path)
    elif path.endswith('.img') or path.endswith('.hdr'):
        return load_analyze(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


# =============================================================================
# Volume Processing Functions
# =============================================================================

def normalize_volume(volume: np.ndarray, percentile_clip: bool = True) -> np.ndarray:
    """Normalize volume to 0-1 range."""
    if volume.max() == 0:
        return volume
    
    if percentile_clip:
        non_zero = volume[volume > 0]
        if len(non_zero) > 0:
            p1, p99 = np.percentile(non_zero, [1, 99])
            volume = np.clip(volume, p1, p99)
    
    vol_min, vol_max = volume.min(), volume.max()
    if vol_max - vol_min > 0:
        volume = (volume - vol_min) / (vol_max - vol_min)
    
    return volume


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Resize 3D volume to target shape."""
    check_dependencies()
    # --- 关键修复：确保输入是 3D ---
    # 使用 np.squeeze 移除所有长度为 1 的冗余维度
    volume = np.squeeze(volume)
    
    # 如果经过 squeeze 还是 4D，强制取前三个维度
    if len(volume.shape) > 3:
        volume = volume[:, :, :, 0]
    # ----------------------------

    # 计算缩放因子，现在的 volume.shape 保证是长度为 3 的序列
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    
    # 执行 3D 缩放
    return ndimage.zoom(volume, factors, order=1)


def extract_slices(
    volume: np.ndarray,
    num_slices: int = 20,
    axis: int = 2,
    margin: float = 0.2
) -> List[np.ndarray]:
    """Extract evenly-spaced 2D slices from 3D volume."""
    depth = volume.shape[axis]
    start = int(depth * margin)
    end = int(depth * (1 - margin))
    
    if start >= end:
        start, end = 0, depth
    
    indices = np.linspace(start, end - 1, num_slices, dtype=int)
    
    slices = []
    for idx in indices:
        if axis == 0:
            s = volume[idx, :, :]
        elif axis == 1:
            s = volume[:, idx, :]
        else:
            s = volume[:, :, idx]
        slices.append(s)
    
    return slices


def resize_slice(slice_2d: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize 2D slice."""
    # --- 关键修复：确保输入是 2D ---
    # np.squeeze 会移除所有长度为 1 的维度 (例如把 (176, 208, 1) 变成 (176, 208))
    slice_2d = np.squeeze(slice_2d)
    
    # 如果经过 squeeze 还是 3D (说明可能切片逻辑有误)，强制取第一层
    if len(slice_2d.shape) > 2:
        slice_2d = slice_2d[:, :, 0]
    # ----------------------------

    if slice_2d.shape[0] == 0 or slice_2d.shape[1] == 0:
        return np.zeros(target_size, dtype=np.float32)
    
    factors = (target_size[0] / slice_2d.shape[0], target_size[1] / slice_2d.shape[1])
    
    # 现在的 factors (长度2) 就能完美匹配 slice_2d (维度2) 了
    return ndimage.zoom(slice_2d, factors, order=1)

def extract_center_slice(volume: np.ndarray, axis: int = 2) -> np.ndarray:
    """Extract the center slice from a volume."""
    mid = volume.shape[axis] // 2
    if axis == 0:
        return volume[mid, :, :]
    elif axis == 1:
        return volume[:, mid, :]
    else:
        return volume[:, :, mid]


# =============================================================================
# CDR Label Conversion
# =============================================================================

def get_cdr_label(cdr_value: float) -> int:
    """Convert CDR value to class label (0-3)."""
    if cdr_value == 0:
        return 0  # NonDemented
    elif cdr_value == 0.5:
        return 1  # VeryMildDemented
    elif cdr_value == 1:
        return 2  # MildDemented
    else:
        return 3  # ModerateDemented


# =============================================================================
# OASIS Dataset Preparation
# =============================================================================

def find_oasis_image(subject_dir: Path) -> Optional[Path]:
    """Find processed MRI image in OASIS subject directory."""
    search_paths = [
        subject_dir / "PROCESSED" / "MPRAGE" / "T88_111",
        subject_dir / "PROCESSED" / "MPRAGE" / "SUBJ_111",
        subject_dir / "PROCESSED" / "MPRAGE",
        subject_dir / "mri",
        subject_dir,
    ]
    
    patterns = ["*masked_gfc.img", "*_111_t88_*.img", "*brain*.img", "*.img", "*.nii.gz", "*.nii"]
    
    for path in search_paths:
        if not path.exists():
            continue
        for pattern in patterns:
            files = list(path.glob(pattern))
            if files:
                return files[0]
    
    return None


def find_all_oasis_subjects(data_dir: Path) -> Dict[str, Path]:
    """Find all OASIS subject directories across all discs."""
    subject_dirs = {}
    
    # Check for disc subdirectories
    disc_dirs = sorted(data_dir.glob('disc*'))
    
    if disc_dirs:
        print(f"Found {len(disc_dirs)} disc directories")
        for disc_dir in disc_dirs:
            for subject_dir in disc_dir. glob('OAS1_*'):
                if subject_dir.is_dir():
                    subject_dirs[subject_dir.name] = subject_dir
    else:
        for subject_dir in data_dir. glob('OAS1_*'):
            if subject_dir.is_dir():
                subject_dirs[subject_dir.name] = subject_dir
    
    return subject_dirs


def find_oasis_csv(data_dir: Path) -> Optional[Path]:
    """Find OASIS CSV file."""
    search_paths = [
        data_dir / "oasis_cross-sectional.csv",
        data_dir. parent / "oasis_cross-sectional.csv",
        Path("./oasis_cross-sectional.csv"),
        Path("./data/oasis_cross-sectional.csv"),
    ]
    
    for search_dir in [data_dir, data_dir. parent, Path("./"), Path("./data")]:
        if search_dir.exists():
            for csv_file in search_dir. glob("*oasis*.csv"):
                search_paths.append(csv_file)
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def prepare_oasis_2d(
    data_dir: str,
    output_dir: str,
    num_slices: int = 20,
    target_size: Tuple[int, int] = (224, 224),
    test_split: float = 0.2
) -> Dict[str, int]:
    """Prepare OASIS dataset as 2D slices."""
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("OASIS 2D PREPARATION")
    print("=" * 60)
    
    # Find CSV
    csv_path = find_oasis_csv(data_dir)
    if csv_path is None:
        raise FileNotFoundError("OASIS CSV not found.  Download from https://www.oasis-brains.org/")
    
    print(f"CSV:  {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['CDR'])
    print(f"Subjects with CDR: {len(df)}")
    
    # Find subjects
    subject_dirs = find_all_oasis_subjects(data_dir)
    print(f"Subject directories: {len(subject_dirs)}")
    
    # Match subjects
    matched = []
    for _, row in df.iterrows():
        sid = row['ID']
        if sid in subject_dirs:
            matched.append({'id': sid, 'dir': subject_dirs[sid], 'cdr': row['CDR']})
    
    print(f"Matched subjects: {len(matched)}")
    
    if len(matched) == 0:
        raise ValueError("No subjects matched!")
    
    # Create directories
    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    for split in ['train', 'test']: 
        for cn in class_names:
            (output_dir / split / cn).mkdir(parents=True, exist_ok=True)
    
    # Split
    try:
        train_subj, test_subj = train_test_split(
            matched, test_size=test_split, stratify=[s['cdr'] for s in matched], random_state=42
        )
    except: 
        train_subj, test_subj = train_test_split(matched, test_size=test_split, random_state=42)
    
    class_counts = {cn: 0 for cn in class_names}
    
    for split_name, subjects in [('train', train_subj), ('test', test_subj)]:
        print(f"\nProcessing {split_name} ({len(subjects)} subjects)...")
        
        for subj in tqdm(subjects, desc=split_name):
            img_path = find_oasis_image(subj['dir'])
            if img_path is None:
                continue
            
            try:
                volume = load_mri(str(img_path))
                volume = normalize_volume(volume)
                slices = extract_slices(volume, num_slices=num_slices, axis=2)
                
                label = get_cdr_label(subj['cdr'])
                class_name = class_names[label]
                
                for i, s in enumerate(slices):
                    s_resized = resize_slice(s, target_size)
                    s_uint8 = (s_resized * 255).astype(np.uint8)
                    img = Image.fromarray(s_uint8, mode='L').convert('RGB')
                    
                    save_path = output_dir / split_name / class_name / f"{subj['id']}_slice{i: 02d}.png"
                    img.save(save_path)
                    class_counts[class_name] += 1
                    
            except Exception as e:
                print(f"Error processing {subj['id']}: {e}")
    
    print(f"\nComplete! Class distribution: {class_counts}")
    return class_counts


def prepare_oasis_3d(
    data_dir: str,
    output_dir: str,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    test_split: float = 0.2
) -> Dict[str, int]:
    """Prepare OASIS dataset as 3D volumes (saved as . npy files)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("OASIS 3D PREPARATION")
    print("=" * 60)
    
    # Find CSV
    csv_path = find_oasis_csv(data_dir)
    if csv_path is None:
        raise FileNotFoundError("OASIS CSV not found.")
    
    print(f"CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['CDR'])
    print(f"Subjects with CDR: {len(df)}")
    
    subject_dirs = find_all_oasis_subjects(data_dir)
    print(f"Subject directories: {len(subject_dirs)}")
    
    matched = []
    for _, row in df.iterrows():
        sid = row['ID']
        if sid in subject_dirs:
            matched.append({'id': sid, 'dir': subject_dirs[sid], 'cdr': row['CDR']})
    
    print(f"Matched subjects: {len(matched)}")
    
    if len(matched) == 0:
        raise ValueError("No subjects matched!")
    
    class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    for split in ['train', 'test']:
        for cn in class_names:
            (output_dir / split / cn).mkdir(parents=True, exist_ok=True)
    
    try:
        train_subj, test_subj = train_test_split(
            matched, test_size=test_split, stratify=[s['cdr'] for s in matched], random_state=42
        )
    except:
        train_subj, test_subj = train_test_split(matched, test_size=test_split, random_state=42)
    
    class_counts = {cn: 0 for cn in class_names}
    
    for split_name, subjects in [('train', train_subj), ('test', test_subj)]:
        print(f"\nProcessing {split_name} ({len(subjects)} subjects)...")
        
        for subj in tqdm(subjects, desc=split_name):
            img_path = find_oasis_image(subj['dir'])
            if img_path is None: 
                continue
            
            try:
                volume = load_mri(str(img_path))
                volume = normalize_volume(volume)
                volume = resize_volume(volume, target_shape)
                
                label = get_cdr_label(subj['cdr'])
                class_name = class_names[label]
                
                save_path = output_dir / split_name / class_name / f"{subj['id']}.npy"
                np.save(save_path, volume. astype(np.float32))
                class_counts[class_name] += 1
                
            except Exception as e:
                print(f"Error processing {subj['id']}: {e}")
    
    print(f"\nComplete!  Class distribution: {class_counts}")
    return class_counts


# =============================================================================
# Kaggle NIfTI Dataset Preparation
# =============================================================================
 
def prepare_kaggle_nifti_2d(
    data_dir: str,
    output_dir: str,
    num_slices: int = 20,
    target_size: Tuple[int, int] = (224, 224),
    test_split: float = 0.2
) -> Dict[str, int]:
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split
    import glob
    import re

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    print("\n" + "=" * 60)
    print("KAGGLE NIfTI -> 2D PREPARATION (FINAL ROBUST VERSION)")
    print("=" * 60)

    # 1. 加载 CSV 并确定名单
    csv_path = Path("oasis_cross-sectional.csv")
    if not csv_path.exists():
        csv_path = Path("data/oasis_cross-sectional.csv")
    
    df = pd.read_csv(csv_path).dropna(subset=['CDR'])
    
    # 2. 按照受试者 ID 进行 Train/Test 划分
    train_df, test_df = train_test_split(df, test_size=test_split, stratify=df['CDR'], random_state=42)
    train_ids = set(train_df['ID'].values)
    test_ids = set(test_df['ID'].values)

    class_mapping = {
        'NonDemented': 'NonDemented', 'Non Demented': 'NonDemented',
        'VeryMildDemented': 'VeryMildDemented', 'Very mild Dementia': 'VeryMildDemented',
        'MildDemented': 'MildDemented', 'Mild Dementia': 'MildDemented',
        'ModerateDemented': 'ModerateDemented', 'Moderate Dementia': 'ModerateDemented'
    }
    
    class_counts = {cn: 0 for cn in ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']}
    nifti_files = list(data_dir.rglob('*.nii')) + list(data_dir.rglob('*.nii.gz'))

    processed_count = 0
    for nifti_path in tqdm(nifti_files, desc="Processing"):
        filename = nifti_path.name
        
        # 3. 正则提取 ID (如 OAS1_0001)
        match = re.search(r"OAS1_\d{4}", filename)
        if not match: continue
        core_id = match.group()
        
        # 4. 决定归属 (基于 CSV 名单)
        current_split = None
        matched_full_id = None
        for tid in train_ids:
            if core_id in tid: current_split = "train"; matched_full_id = tid; break
        if not current_split:
            for tid in test_ids:
                if core_id in tid: current_split = "test"; matched_full_id = tid; break
        
        if not current_split: continue

        # 5. 匹配标签 (搜索 data/Kaggle 图片库)
        target_class = None
        search_pattern = os.path.join("data", "Kaggle", "**", f"*{core_id}*")
        matching_images = glob.glob(search_pattern, recursive=True)

        if matching_images:
            parent_name = Path(matching_images[0]).parent.name
            clean_parent = parent_name.lower().replace(" ", "")
            for key, value in class_mapping.items():
                if key.lower().replace(" ", "") in clean_parent:
                    target_class = value; break
        
        if target_class is None: continue

        # 6. 处理并保存
        save_dir = output_dir / current_split / target_class
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            volume = load_nifti(str(nifti_path))
            volume = normalize_volume(volume)
            slices = extract_slices(volume, num_slices=num_slices, axis=2)
            for i, s in enumerate(slices):
                s_resized = resize_slice(s, target_size)
                img = Image.fromarray((s_resized * 255).astype(np.uint8), mode='L').convert('RGB')
                img.save(save_dir / f"{matched_full_id}_slice{i:02d}.png")
                class_counts[target_class] += 1
            processed_count += 1
        except Exception as e:
            print(f"Error processing {matched_full_id}: {e}")

    print(f"Successfully processed {processed_count} subjects into 2D slices")
    return class_counts

def prepare_kaggle_nifti_3d(
    data_dir: str,
    output_dir: str,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    test_split: float = 0.2
) -> Dict[str, int]:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import glob
    import re

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    print("\n" + "=" * 60)
    print("KAGGLE NIfTI → 3D PREPARATION (FINAL VERSION)")
    print("=" * 60)

    # 1. 加载并寻找 CSV
    csv_path = find_oasis_csv(data_dir) or find_oasis_csv(data_dir.parent) or Path("data/oasis/oasis_cross-sectional.csv")
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}"); return {}

    # 关键：只有带 CDR 标签的人才会被加入 split 名单
    df = pd.read_csv(csv_path).dropna(subset=['CDR'])
    train_df, test_df = train_test_split(df, test_size=test_split, stratify=df['CDR'], random_state=42)
    train_ids, test_ids = set(train_df['ID'].values), set(test_df['ID'].values)

    class_mapping = {
        'NonDemented': 'NonDemented', 'Non Demented': 'NonDemented',
        'VeryMildDemented': 'VeryMildDemented', 'Very mild Dementia': 'VeryMildDemented',
        'MildDemented': 'MildDemented', 'Mild Dementia': 'MildDemented',
        'ModerateDemented': 'ModerateDemented', 'Moderate Dementia': 'ModerateDemented'
    }
    
    class_counts = {cn: 0 for cn in ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']}
    nifti_files = list(data_dir.rglob('*.nii')) + list(data_dir.rglob('*.nii.gz'))

    processed_count = 0
    for nifti_path in tqdm(nifti_files, desc="3D Processing"):
        # 2. 激进提取 ID
        match = re.search(r"OAS1_\d{4}", nifti_path.name)
        if not match: continue
        core_id = match.group()
        
        # 3. 匹配 Split (train/test)
        current_split = None
        matched_full_id = None
        for tid in train_ids:
            if core_id in tid: current_split = "train"; matched_full_id = tid; break
        if not current_split:
            for tid in test_ids:
                if core_id in tid: current_split = "test"; matched_full_id = tid; break
        
        if not current_split: continue

        # 4. 匹配标签 (使用 core_id 映射 Kaggle 目录)
        target_class = None
        search_pattern = os.path.join("data", "Kaggle", "**", f"*{core_id}*")
        matching_imgs = glob.glob(search_pattern, recursive=True)
        if matching_imgs:
            parent_name = Path(matching_imgs[0]).parent.name
            clean_parent = parent_name.lower().replace(" ", "")
            for k, v in class_mapping.items():
                if k.lower().replace(" ", "") in clean_parent:
                    target_class = v; break
        
        if target_class is None: continue

        # 5. 执行处理
        save_dir = output_dir / current_split / target_class
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            volume = load_nifti(str(nifti_path))
            volume = normalize_volume(volume)
            volume = resize_volume(volume, target_shape) # 包含 squeeze 修复
            np.save(save_dir / f"{matched_full_id}.npy", volume.astype(np.float32))
            class_counts[target_class] += 1
            processed_count += 1
        except Exception as e:
            print(f"Error at {core_id}: {e}")

    print(f"Successfully processed {processed_count} subjects into 3D volumes.")
    return class_counts

def split_original_kaggle_images(kaggle_dir: str, output_dir: str, test_split: float = 0.2):
    """
    Split original Kaggle JPG images into train/test based on CSV labels.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import shutil
    import re

    kaggle_dir, output_dir = Path(kaggle_dir), Path(output_dir)
    
    # Load CSV to get subject list
    csv_path = Path("oasis_cross-sectional.csv")
    if not csv_path.exists():
        csv_path = Path("data/oasis_cross-sectional.csv")
    
    # Filter subjects with CDR labels
    df = pd.read_csv(csv_path).dropna(subset=['CDR'])
    
    # Split subjects into train/test
    train_df, test_df = train_test_split(df, test_size=test_split, stratify=df['CDR'], random_state=42)
    train_ids = set(train_df['ID'].values)
    test_ids = set(test_df['ID'].values)

    print(f"\nSorting JPGs from {kaggle_dir} to {output_dir}...")
    
    jpg_files = list(kaggle_dir.rglob('*.jpg'))
    for jpg_path in tqdm(jpg_files, desc="Copying"):
        match = re.search(r"OAS1_\d{4}", jpg_path.name)
        if not match: continue
        core_id = match.group()
        
        # Check split membership
        split = None
        if any(core_id in tid for tid in train_ids):
            split = "train"
        elif any(core_id in tid for tid in test_ids):
            split = "test"
            
        if not split: continue
        
        # Normalize class folder names
        target_class = jpg_path.parent.name.replace(" ", "")
        #if "Dementia" in target_class:
        #    target_class = target_class.replace("Dementia", "Demented")
        #if target_class == "NonDemented": pass 
        dest = output_dir / split / target_class
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jpg_path, dest / jpg_path.name)

    print(f"Done! Check your split data in: {output_dir}")

if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess MRI data')
    parser.add_argument('--dataset', type=str, choices=['oasis', 'kaggle_nifti','kaggle_orig'], required=True)
    parser.add_argument('--mode', type=str, choices=['2d', '3d'], default='2d')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_slices', type=int, default=20)
    
    args = parser.parse_args()
    
    if args.dataset == 'oasis': 
        if args.mode == '2d': 
            prepare_oasis_2d(args.data_dir, args.output_dir, args.num_slices)
        else: 
            prepare_oasis_3d(args.data_dir, args.output_dir)
    elif args.dataset == 'kaggle_nifti':
        if args.mode == '2d':
            prepare_kaggle_nifti_2d(args.data_dir, args.output_dir, args.num_slices)
        else:
            prepare_kaggle_nifti_3d(args.data_dir, args.output_dir)
    elif args.dataset == 'kaggle_orig':
        split_original_kaggle_images(args.data_dir, args.output_dir)