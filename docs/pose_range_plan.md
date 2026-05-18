# pose_range 透传修改计划

**Goal:** 将 `pose_targets.py` 的坐标映射从 `clamp(0,1)` 改为基于 `pose_range=(-0.8, 0.8)` 的线性映射，并在 `train.py` 和 `models/wiflow_model.py` 中同步透传。

**公式：**
- 编码: `heatmap_coord = (x - pose_min) / (pose_max - pose_min) * (H - 1)`
- 解码: `x = heatmap_coord / (H - 1) * (pose_max - pose_min) + pose_min`

---

## 1. `pose_targets.py` — `keypoints_to_heatmap_coords`

```python
def keypoints_to_heatmap_coords(
    keypoints: torch.Tensor,
    heatmap_size: int,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """Map H36M-17 keypoints from pose_range to heatmap coordinates."""
    if keypoints.ndim != 3 or keypoints.shape[-2:] != (NUM_H36M_KEYPOINTS, 2):
        raise ValueError(f"Expected keypoints shaped [B, 17, 2], got {tuple(keypoints.shape)}")
    if heatmap_size < 2:
        raise ValueError("heatmap_size must be at least 2")
    pose_min, pose_max = pose_range
    span = pose_max - pose_min
    return (keypoints - pose_min) / span * float(heatmap_size - 1)
```

## 2. `pose_targets.py` — `build_pcm_targets`

- 新增 `pose_range: tuple[float, float] = (-0.8, 0.8)` 参数
- 透传至 `keypoints_to_heatmap_coords()`
- docstring 更新

## 3. `pose_targets.py` — `build_paf_targets`

- 新增 `pose_range: tuple[float, float] = (-0.8, 0.8)` 参数
- 透传至 `keypoints_to_heatmap_coords()`
- docstring 更新

## 4. `pose_targets.py` — `build_pcm_paf_targets`

- 新增 `pose_range: tuple[float, float] = (-0.8, 0.8)` 参数
- 透传至 `build_pcm_targets()` 和 `build_paf_targets()`

## 5. `pose_targets.py` — `decode_pcm_argmax`

- 新增 `pose_range: tuple[float, float] = (-0.8, 0.8)` 参数
- 解码反算：`(heatmap_coord / (H_or_W - 1)) * span + pose_min`
- docstring 更新

## 6. `train.py` — `compute_losses`

- 新增 `pose_range: tuple[float, float] = (-0.8, 0.8)` 参数
- 调用 `build_pcm_paf_targets()` 时传入 `pose_range=pose_range`
- `run_training()` 中调用 `run_epoch()` 时不需要传（用默认值）

## 7. `models/wiflow_model.py` — `decode_features`

- `__init__` 新增 `pose_range: tuple[float, float] = (-0.8, 0.8)` 参数，存储为 `self.pose_range`
- `decode_features()` 中调用 `decode_pcm_argmax()` 时传入 `pose_range=self.pose_range`
- docstring 更新
