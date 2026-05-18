# build_memmap.py 清理与 GT 数据流修正

> **破坏性变更** — 本次只修改 `build_memmap.py`。`data/memmap_dataset.py` 和 `dataloader.py` 将在后续 PR 中适配 H36M-17 格式。

**Goal:** 删除 OpenPose18 相关逻辑，raw RGB reference 不映射不归一化，real GT 只保留 xy 通道。

---

## 1. 删除 OpenPose18 相关逻辑

删除以下符号和所有引用：
- `COCO17_TO_OPENPOSE18`
- `_valid_point`
- `coco17_to_openpose18`
- `normalize_kpts_to_pose_range`
- 所有以 `kpts18` 命名的变量

## 2. 保留 `rgb/` 读取但只存原始 17 点

- `rgb/frame*.npy` → `reference_keypoints`，不做关节映射，不归一化
- 保持 COCO17 原始 17 点格式，作为对齐参考，**完全不参与训练**

## 3. 修改 `process_trial`

输出：
- `csi` (CSI 帧)
- `reference_keypoints` (raw COCO17 17点, (N,17,2))
- `environment / sample / action / frame_idx`

不再输出 `kpts18`。

调用 `normalize_kpts_to_pose_range` 的地方改为直接 `np.load` 后保持原样。

## 4. 修改拼接逻辑

- 删除 `all_kpts18`
- 新增 `all_reference_keypoints`，shape 为 `(N_total, 17, 2)`

## 5. 修改 GT 读取逻辑

- 从 `--gt-dir` 读取 H36M-17 GT 文件 `(F, 17, 3)`
- 第三通道为 confidence，丢弃
- 只保留 xy → `ground_truth.npy`，shape `(N_total, 17, 2)`
- **不保存 conf，不保存 mask**

## 6. 对齐逻辑

GT 文件 frame001-297 连续无缺，当前帧数断言 + 顺序拼接已保证 action+subject+frame_idx 对齐。无需额外对齐逻辑。

## 7. 修改保存文件

保存：
- `reference_keypoints.npy` — COCO17 raw 17 点 `(N, 17, 2)`
- `ground_truth.npy` — H36M-17 `(N, 17, 2)`
- `csi_gminmax.npy` / `csi_gzscore.npy` / `csi_zscore.npy` — 不变
- `meta.npz` — 不变

不再保存 OpenPose18 格式数据。

## 8. 修改 `stats.json`

新增：
- `pose_format: "H36M17"`
- `reference_format: "raw_coco17_no_mapping"`
- `ground_truth_shape: [N, 17, 2]`
- `reference_keypoints_shape: [N, 17, 2]`

删除或标记为 unused：
- `pose_min` / `pose_max`（reference 不再归一化，GT 已预归一化到 [-0.8, 0.8]）

## 9. 修改顶部 docstring

移除 COCO17/OpenPose18 描述，改为：
```
Input:
    /data/WiFiPose/dataset/dataset/{ACTION}/{SUBJECT}/
        wifi-csi/frame*.mat   ← CSIamp (3, 114, 10)
        rgb/frame*.npy        ← COCO17 keypoints (17, 2) — reference only

Output:
    csi_gminmax.npy          ← (N, 64, 3, 114)
    csi_gzscore.npy          ← (N, 64, 3, 114)
    csi_zscore.npy           ← (N, 64, 3, 114)
    ground_truth.npy         ← H36M-17 GT (N, 17, 2)
    reference_keypoints.npy  ← raw COCO17 (N, 17, 2) — not for training
    meta.npz
    stats.json
```
