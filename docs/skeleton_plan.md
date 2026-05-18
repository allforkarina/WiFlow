# models/skeleton.py + models/__init__.py 修改计划

> **破坏性变更** — `wiflow_joint_decoder.py`, `wiflow_hierarchical_joint_decoder.py`, `wiflow_heatmap_decoder.py`, `train.py`, `eval.py`, `pose_targets.py` 将在后续依次修复。

**Goal:** 将 skeleton 拓扑从 OpenPose18 (18 joints / 19 edges) 切换到 H36M-17 (17 joints / 16 edges)，同步更新导出层。

---

## 1. `models/skeleton.py` — 删除 OpenPose18

- 删除 `NUM_OPENPOSE_KEYPOINTS = 18`
- 删除 `OPENPOSE_BONE_EDGES` (19 条边)

## 2. `models/skeleton.py` — 新增 H36M-17

- `NUM_H36M_KEYPOINTS = 17`

- `H36M17_NAMES`:
  ```python
  H36M17_NAMES = [
      "pelvis", "r_hip", "r_knee", "r_ankle",
      "l_hip", "l_knee", "l_ankle",
      "spine", "thorax", "neck", "head",
      "l_shoulder", "l_elbow", "l_wrist",
      "r_shoulder", "r_elbow", "r_wrist",
  ]
  ```

- `H36M_BONE_EDGES` (16 条边):
  ```python
  H36M_BONE_EDGES: tuple[tuple[int, int], ...] = (
      (0, 1), (1, 2), (2, 3),       # right leg
      (0, 4), (4, 5), (5, 6),       # left leg
      (0, 7), (7, 8), (8, 9), (9, 10),  # spine → head
      (8, 11), (11, 12), (12, 13),  # left arm
      (8, 14), (14, 15), (15, 16),  # right arm
  )
  ```

## 3. `models/skeleton.py` — 修改 `build_normalized_adjacency`

- 默认 `num_nodes=NUM_H36M_KEYPOINTS`
- 默认 `edges=H36M_BONE_EDGES`
- docstring 改为 "H36M-17 keypoints"

## 4. `models/skeleton.py` — 更新文件注释

- 描述改为 H36M-17 skeleton topology

## 5. `models/__init__.py` — 更新导入和导出

- 导入: `NUM_OPENPOSE_KEYPOINTS, OPENPOSE_BONE_EDGES` → `NUM_H36M_KEYPOINTS, H36M_BONE_EDGES, H36M17_NAMES`
- `__all__` 中对应替换: 删除旧名, 新增 `"NUM_H36M_KEYPOINTS"`, `"H36M_BONE_EDGES"`, `"H36M17_NAMES"`
- `build_normalized_adjacency` 保留
