# memmap_dataset.py 修改计划

> **破坏性变更** — `dataloader.py` 中 `batch["kpts18"]` 将在后续 PR 同步改为 `batch["keypoints"]`。

**Goal:** 将数据集从 OpenPose18 迁移到 H36M-17，新增可选 reference 加载，软关闭 PCM/PAF。

---

## 1. 修改文件顶部 docstring

将 OpenPose18 描述改为 H36M-17 GT，标明 `reference_keypoints.npy` 仅作参考不参与训练。

## 2. 修改变量命名

- `self._kpts18` → `self._keypoints`
- 局部变量 `kpts18` → `keypoints`

## 3. GT 数据语义

- 继续读取 `ground_truth.npy`
- 语义改为 H36M-17，shape `(N, 17, 2)`

## 4. 新增可选 reference 加载

- `__init__` 新增参数 `load_reference: bool = False`
- 为 `True` 时加载 `reference_keypoints.npy` → `self._reference_keypoints`
- 不作为训练 label

## 5. 修改 `__getitem__` 返回字段

- 删除 `"kpts18"`
- 新增 `"keypoints"`（来自 `ground_truth.npy`，H36M-17）
- 当 `load_reference=True` 时新增 `"reference_keypoints"`

## 6-7. 软关闭 PCM/PAF

- 注释掉 `from data.heatmap_gt import build_pcm_paf`
- `build_targets` 默认值改为 `False`
- 注释掉 `if self.build_targets:` 分支内的 PCM/PAF 生成逻辑
- 保留 `build_targets` 参数以便后续恢复

## 8. 保留不动

- `_build_split` / `CSI_FILES` / `meta.npz` 读取逻辑全部保留
- `__init__` 其余参数不变
