import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import create_data_loader
from models.wiflow_model import WiFlowModel
from pose_targets import build_pcm_targets
from train import compute_losses, prepare_model_input


def diagnose_data_and_loss(h5_path: str):
    print(f"=== Diagnosing Data and Initial Loss ===")
    print(f"Dataset Path: {h5_path}")
    
    # 1. Initialize dataloader
    # We just need a single batch, so split="train", batch_size=8
    loader = create_data_loader(
        dataset_root=h5_path,
        split="train",
        batch_size=8,
        shuffle=False
    )
    
    # Get one batch
    batch = next(iter(loader))
    
    # 2. Inspect keypoints bounds
    keypoints = torch.as_tensor(batch["keypoints"], dtype=torch.float32)
    kpts0 = keypoints[0] # First sample in batch
    print("\n--- 1. Keypoints Coordinate Inspection ---")
    print(f"Keypoints shape: {keypoints.shape}")
    print(f"Sample 0 keypoints X range: [{kpts0[:, 0].min().item():.4f}, {kpts0[:, 0].max().item():.4f}]")
    print(f"Sample 0 keypoints Y range: [{kpts0[:, 1].min().item():.4f}, {kpts0[:, 1].max().item():.4f}]")
    
    is_in_zero_one = (kpts0.min() >= 0.0) and (kpts0.max() <= 1.0)
    is_in_neg_pos = (kpts0.min() >= -1.0) and (kpts0.max() <= 1.0) and (kpts0.min() < 0.0)
    
    if is_in_zero_one:
        print("-> Range looks like [0, 1]")
    elif is_in_neg_pos:
        print("-> Range looks like [-1, 1] or [-0.8, 0.8]")
    else:
        print("-> Range is outside expected normalized bounds!")

    # 3. Inspect PCM (Part Confidence Maps) targets
    print("\n--- 2. PCM Target Inspection ---")
    heatmap_size = 36
    sigma = 1.5
    pcm_targets = build_pcm_targets(keypoints, heatmap_size=heatmap_size, sigma=sigma)
    print(f"PCM shape: {pcm_targets.shape}")
    
    nose_pcm = pcm_targets[0, 0] # Sample 0, channel 0 (Nose)
    nose_max = nose_pcm.max().item()
    print(f"Nose channel maximum peak value: {nose_max:.4f}")
    if nose_max > 0.5:
        print(f"-> VALID: Nose channel has a clear peak (> 0.5)")
    else:
        print(f"-> WARNING: Nose channel peak is very low or nonexistent! Is the keypoint far out of bounds?")
    
    # Inspect [0,0] occlusion handling symptom
    # If [0,0] keypoints fall back to (0,0) in heatmap, the peak is exactly at the top-left index
    origin_peak = nose_pcm[0, 0].item()
    print(f"Nose channel value exactly at top-left origin (0, 0): {origin_peak:.4f}")

    # 4. Check initial model loss
    print("\n--- 3. Initial Loss Validation ---")
    
    csi_features = ["csi_amplitude", "csi_phase_cos"]
    model_input, target_keypoints = prepare_model_input(batch, device=torch.device("cpu"), csi_features=csi_features)
    print(f"Model input shape (CSI): {model_input.shape}")
    
    # Instantiate Model
    # 2 features * 3 antennas = 6 channels
    model = WiFlowModel(
        csi_channels=len(csi_features) * 3,
        decoder_type="heatmap_msfn",
        heatmap_size=heatmap_size
    )
    model.eval() # Prevent batchnorm updates during diagnosis
    
    with torch.no_grad():
        prediction = model(model_input)
    
    losses = compute_losses(
        prediction=prediction,
        target=target_keypoints,
        heatmap_size=heatmap_size,
        heatmap_sigma=sigma,
        paf_width=1.0,
        paf_loss_weight=1.0
    )
    
    print("Initial Output Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose keypoint ranges, PCM generation, and initial loss.")
    parser.add_argument("--h5-path", type=str, default="/data/WiFiPose/dataset/mmfi_pose.h5", help="Path to HDF5 dataset")
    args = parser.parse_args()
    
    try:
        diagnose_data_and_loss(args.h5_path)
    except Exception as e:
        print(f"\nExecution Failed: {e}")


if __name__ == "__main__":
    main()
