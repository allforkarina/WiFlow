# 进度日志

## 会话：2026-05-12 — 训练速度诊断

### 阶段 1：瓶颈诊断与量化分析
- **状态：** complete
- 执行的操作：
  - 完整审查了 train.py 的训练循环和数据流
  - 审查了 dataloader.py 的 MMFiPoseDataset.__getitem__ 和 create_data_loader
  - 审查了 model 的三个核心组件（SpatialEncoder, AxialEncoder, JointDecoder）
  - 量化分析：192K train + 64K val 帧，batch_size=64 → ~4000 batches/epoch
  - 每 batch ~120ms，其中估计 60-80ms 花在 HDF5 随机读取上
  - 根因确认：num_workers=0 + per-sample HDF5 seek 导致 GPU 利用率仅 ~42%

### 阶段 2：假设验证
- **状态：** complete
- 执行的操作：
  - 用户确认 H1 方案
  - 按计划实施 num_workers 优化

### 阶段 3：实施优化 — H1
- **状态：** complete
- 执行的操作：
  - TrainConfig.num_workers 默认值: 0 → 4
  - argparse --num-workers 默认值: 0 → 4
  - DataLoader 添加 `pin_memory=True`
  - DataLoader 添加 `persistent_workers=True`（num_workers > 0 时）
- 创建/修改的文件：
  - train.py: L49, L488（num_workers 默认值 0→4）
  - dataloader.py: L843-844（+pin_memory=True, +persistent_workers）

### 阶段 4：验证与测试
- **状态：** pending
- **说明：** 需用户在 Linux 服务器上实际运行验证

### 阶段 5：交付
- **状态：** pending

## 修改摘要

| 文件 | 行 | 改动 |
|------|-----|------|
| train.py | 49 | `num_workers: int = 0` → `4` |
| train.py | 488 | `default=0` → `default=4` |
| dataloader.py | 843 | `)` → `, pin_memory=True, persistent_workers=num_workers > 0)` |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | H1 已完成，等待用户验证 |
| 我要去哪里？ | 用户决定是否继续 H2（批量 HDF5 读取） |
| 目标是什么？ | 将 ~480s/epoch 降低到 ~120-180s/epoch |
| 我学到了什么？ | H1 预期加速 2-3x；实际效果需用户验证 |
| 我做了什么？ | TrainConfig + argparse 默认值改为 4，DataLoader 加 pin_memory + persistent_workers |

---
*每个阶段完成后或遇到错误时更新此文件*
