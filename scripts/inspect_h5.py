import argparse
import h5py
from pathlib import Path


def inspect_h5_file(file_path: str) -> None:
    """Inspect and print all attributes and dataset structures within an HDF5 file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Cannot find file {file_path}")
        return

    print(f"=== HDF5 File Inspection: {file_path} ===\n")
    
    with h5py.File(path, "r") as h5_file:
        # 1. Print global attributes (Metadata)
        print("--- [Attributes] ---")
        if len(h5_file.attrs) == 0:
            print("  (None)")
        else:
            for key, value in h5_file.attrs.items():
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                print(f"  {key}: {value}")
        
        print("\n--- [Datasets] ---")
        # 2. Iterate and print dataset shapes and types
        for key in sorted(h5_file.keys()):
            item = h5_file[key]
            if isinstance(item, h5py.Dataset):
                shape = item.shape
                dtype = item.dtype
                print(f"  • {key}")
                print(f"      Shape: {shape}")
                print(f"      Dtype: {dtype}")
                
                # Preview small index arrays
                if "indices" in key and len(shape) == 1 and shape[0] > 0:
                    preview = item[:5].tolist()
                    print(f"      Preview (first 5): {preview} ...")
                
                # Preview string/metadata arrays
                elif key in ["action", "environment", "sample", "frame_id"]:
                    val = item[0]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    print(f"      Preview (1st item): {val} ...")
            elif isinstance(item, h5py.Group):
                print(f"  • [Group] {key} (contains {len(item.keys())} items)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an HDF5 dataset file")
    parser.add_argument(
        "--h5-path", 
        type=str, 
        default="/data/WiFiPose/dataset/mmfi_pose.h5", 
        help="Path to the HDF5 file (default: /data/WiFiPose/dataset/mmfi_pose.h5)"
    )
    args = parser.parse_args()
    
    inspect_h5_file(args.h5_path)


if __name__ == "__main__":
    main()
