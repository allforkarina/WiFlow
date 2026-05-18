# 收尾修改计划

**Goal:** 清理残留的 OpenPose18 代码，更新诊断工具，增强 eval 输出。

---

## 1. `data/heatmap_gt.py` — 标记 DEPRECATED

- 文件顶部添加 `# DEPRECATED: use pose_targets.py for online H36M-17 PCM/PAF generation`
- 内部代码不动

## 2. `data/memmap_dataset.py` — 删除注释残留

- 删除第 11 行：`# from data.heatmap_gt import build_pcm_paf  # soft-disabled: H36M-17 migration`
- 删除 `__getitem__` 中注释掉的 PCM/PAF 分支（第 157-167 行）
- 删除 `build_targets` 参数和 `self.build_targets` 赋值（已无效）
- 删除 `self.heatmap_size`、`self.heatmap_sigma`、`self.paf_width`、`self.pose_range` 赋值（PCM/PAF 已移除，不再需要）
- `__init__` 签名中删除 `heatmap_size`、`heatmap_sigma`、`paf_width`、`pose_range`、`build_targets` 参数

## 3. `scripts/diagnose_loss.py` — 适配 H36M-17

- 修正 `create_memmap_data_loader(dataset_root=...)` → `data_dir=...`
- 删除 OpenPose/nose 语义，改为检查 pelvis（index 0）
- PCM shape 检查：`[B, 17, H, W]` 而非 `[B, 18, H, W]`
- PAF shape 检查：`[B, 32, H, W]` 而非 `[B, 38, H, W]`
- Label shape 检查：`[B, 17, 2]` 而非 `[B, 18, 2]`
- docstring 更新

## 4. `eval.py` — per-joint CSV 增加名称

- `build_joint_metric_rows` 中新增 `"joint_name"` 字段
- 从 `H36M17_NAMES` 导入并查找
- 不影响其他逻辑

---
