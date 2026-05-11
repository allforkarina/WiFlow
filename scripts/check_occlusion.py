import argparse
import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import dataset discovering tools from the custom dataloader
from dataloader import discover_sample_sequences

def scan_raw_dataset(dataset_root: Path) -> None:
    """Scan raw MM-Fi .npy files to find occlusion [0, 0] targets."""
    print(f"=== Scanning Raw Dataset: {dataset_root} ===")
    
    try:
        sequences = discover_sample_sequences(dataset_root)
    except Exception as e:
        print(f"Error reading root: {e}")
        return

    total_points = 0
    occluded_points = 0
    frames_with_occlusion = 0
    total_frames = 0
    
    for seq in tqdm(sequences, desc="Scanning raw sequences", dynamic_ncols=True):
        npy_files = sorted(seq.rgb_dir.glob("*.npy"))
        for npy_file in npy_files:
            try:
                kpts = np.load(npy_file) # shape should be (17, 2)
            except Exception:
                continue
            
            # Check for coordinates strictly equaling [0, 0]
            is_origin = np.all(kpts == 0.0, axis=-1)
            occs = np.sum(is_origin)
            
            occluded_points += occs
            total_points += kpts.shape[0]
            total_frames += 1
            if occs > 0:
                frames_with_occlusion += 1
                
    print("\n--- Raw Dataset Summary ---")
    print(f"Total Keypoints: {total_points}")
    print(f"Occluded Keypoints [0, 0]: {occluded_points}")
    print(f"Occlusion Percentage: {(occluded_points / max(1, total_points)) * 100:.2f}%")
    print(f"Frames with at least one occlusion: {frames_with_occlusion} / {total_frames}\n")


def scan_h5_dataset(h5_path: Path) -> None:
    """Scan pre-built HDF5 files to find occlusion [0, 0] targets."""
    print(f"=== Scanning HDF5 Dataset: {h5_path} ===")
    
    with h5py.File(h5_path, "r") as f:
        if "keypoints" not in f:
            print("Error: 'keypoints' dataset not found in HDF5 file.")
            return

        keypoints = f["keypoints"][:]
        is_origin = np.all(keypoints == 0.0, axis=-1)
        
        total_points = keypoints.shape[0] * keypoints.shape[1]
        occluded_points = np.sum(is_origin)
        frames_with_occlusion = np.sum(np.any(is_origin, axis=-1))
        total_frames = keypoints.shape[0]
        
        print("\n--- HDF5 Dataset Summary ---")
        print(f"Total Keypoints: {total_points}")
        print(f"Occluded Keypoints [0, 0]: {occluded_points}")
        print(f"Occlusion Percentage: {(occluded_points / max(1, total_points)) * 100:.2f}%")
        print(f"Frames with at least one occlusion: {frames_with_occlusion} / {total_frames}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan MM-Fi dataset raw & h5 files for occluded [0, 0] points.")
    parser.add_argument("--raw-root", type=str, default="/data/WiFiPose/dataset/dataset", help="Path to raw dataset dir")
    parser.add_argument("--h5-path", type=str, default="/data/WiFiPose/dataset/mmfi_pose.h5", help="Path to h5 dataset")
    args = parser.parse_args()
    
    raw_path = Path(args.raw_root)
    h5_path = Path(args.h5_path)
    
    if raw_path.exists():
        scan_raw_dataset(raw_path)
    else:
        print(f"Raw dataset path not found: {raw_path}")
        
    if h5_path.exists():
        scan_h5_dataset(h5_path)
    else:
        print(f"HDF5 dataset path not found: {h5_path}")


if __name__ == "__main__":
    main()
