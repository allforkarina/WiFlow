import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import create_memmap_data_loader
from models.wiflow_model import WiFlowModel
from pose_targets import build_pcm_targets
from train import compute_losses, prepare_model_input


def diagnose_data_and_loss(dataset_dir: str):
    print(f"=== Diagnosing H36M-17 Data and Initial Loss ===")
    print(f"Dataset Path: {dataset_dir}")

    loader = create_memmap_data_loader(
        data_dir=dataset_dir,
        split="train",
        batch_size=8,
        shuffle=False,
    )

    batch = next(iter(loader))

    keypoints = torch.as_tensor(batch["keypoints"], dtype=torch.float32)
    kpts0 = keypoints[0]
    print("\n--- 1. Keypoints Coordinate Inspection ---")
    print(f"Keypoints shape: {keypoints.shape}")
    print(f"Sample 0 keypoints X range: [{kpts0[:, 0].min().item():.4f}, {kpts0[:, 0].max().item():.4f}]")
    print(f"Sample 0 keypoints Y range: [{kpts0[:, 1].min().item():.4f}, {kpts0[:, 1].max().item():.4f}]")

    is_in_zero_one = (kpts0.min() >= 0.0) and (kpts0.max() <= 1.0)
    is_in_pose_range = (kpts0.min() >= -1.0) and (kpts0.max() <= 1.0) and (kpts0.min() < 0.0)

    if is_in_zero_one:
        print("-> Range looks like [0, 1]")
    elif is_in_pose_range:
        print("-> Range looks like [-0.8, 0.8] (pose_range)")
    else:
        print("-> Range is outside expected normalized bounds!")

    print("\n--- 2. PCM Target Inspection ---")
    heatmap_size = 36
    sigma = 1.5
    pcm_targets = build_pcm_targets(keypoints, heatmap_size=heatmap_size, sigma=sigma)
    print(f"PCM shape: {pcm_targets.shape}")

    pelvis_pcm = pcm_targets[0, 0]
    pelvis_max = pelvis_pcm.max().item()
    print(f"Pelvis (index 0) channel maximum peak value: {pelvis_max:.4f}")
    if pelvis_max > 0.5:
        print(f"-> VALID: Pelvis channel has a clear peak (> 0.5)")
    else:
        print(f"-> WARNING: Pelvis channel peak is very low or nonexistent! Is the keypoint far out of bounds?")

    origin_peak = pelvis_pcm[0, 0].item()
    print(f"Pelvis channel value exactly at top-left origin (0, 0): {origin_peak:.4f}")

    print("\n--- 3. Initial Loss Validation ---")

    model_input, target_keypoints = prepare_model_input(batch, device=torch.device("cpu"))
    print(f"Model input shape (CSI): {model_input.shape}")

    model = WiFlowModel(
        input_channels=3,
        decoder_type="heatmap_msfn",
        heatmap_size=heatmap_size,
    )
    model.eval()

    with torch.no_grad():
        prediction = model(model_input)

    losses = compute_losses(
        prediction=prediction,
        target=target_keypoints,
        heatmap_size=heatmap_size,
        heatmap_sigma=sigma,
        paf_width=1.0,
        paf_loss_weight=1.0,
    )

    print("Initial Output Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose keypoint ranges, PCM generation, and initial loss.")
    parser.add_argument("--dataset-dir", type=str, default="data/mmfi_pose", help="Path to NPY memmap dataset directory")
    args = parser.parse_args()

    try:
        diagnose_data_and_loss(args.dataset_dir)
    except Exception as e:
        print(f"\nExecution Failed: {e}")


if __name__ == "__main__":
    main()