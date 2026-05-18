# dataloader.py 修改计划

**Goal:** 将 collate 函数和工厂函数从 OpenPose18 `kpts18` 迁移到 H36M-17 `keypoints`，新增可选 reference 支持。

---

## 1. 修改 `memmap_collate_fn`

- `item["kpts18"]` → `item["keypoints"]`
- 方案 A：检查 `"reference_keypoints" in batch[0]` → 有则 stack 进 `"reference_keypoints"`
- 输出字段：`csi_amplitude`、`keypoints`、`action`、`sample`、`environment`、`frame_idx`，可选 `reference_keypoints`

## 2. 保持输出不变

- `keypoints` shape: `[B, 17, 2]` (H36M-17)
- reference shape: `[B, 17, 2]` (raw COCO17)
- `csi.permute(0, 2, 3, 1)` 不动

## 3. 可选 reference collate

```python
if "reference_keypoints" in batch[0]:
    ref = torch.stack([item["reference_keypoints"] for item in batch])
    result["reference_keypoints"] = ref
```

## 4. 工厂函数新增 `load_reference`

- `create_memmap_data_loader()` 新增参数 `load_reference: bool = False`
- 传给 `MemmapDataset(load_reference=load_reference, ...)`
- `create_memmap_data_loaders()` 新增同名参数，透传给 `create_memmap_data_loader()`

## 5. 保持不变

- `csi.permute(0, 2, 3, 1)` 不动
- `build_targets=False` 不动
- split/shuffle/worker/pin_memory 逻辑不动

## 6. 修改 preview 输出

- `kpts18` → `keypoints`
- 当 `load_reference=True` 时打印 `reference_keypoints` shape

## 7. 修改注释

- 删除 OpenPose18/kpts18 描述
- 改为 H36M17 keypoints
