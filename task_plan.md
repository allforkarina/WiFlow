# 任务计划：训练速度优化 — 定位并修复 ~480s/epoch 的性能瓶颈

## 目标
将单 epoch 训练时间从 ~480s 降低到 ~120-180s（目标 2.5-4x 加速），主要通过优化数据加载流水线。

## 当前阶段
阶段 2 — 假设验证

## 各阶段

### 阶段 1：瓶颈诊断与量化分析
- [x] 分析训练循环时间构成
- [x] 定位主要瓶颈：数据加载（num_workers=0 + HDF5 随机读取）
- [x] 量化每个 batch 的时间分布
- **状态：** complete

### 阶段 2：假设验证
- [ ] 确认假设优先级排序
- [ ] 用户确认修改方案
- **状态：** in_progress

### 阶段 3：实施优化
- [ ] H1 — 启用多进程数据加载（num_workers + pin_memory）
- [ ] H2 — 批量 HDF5 读取（自定义 collate_fn，减少 seek 次数）
- [ ] H3 — 减少 epoch 级 I/O 开销（条件化 checkpoint 保存、减少 val 频率）
- [ ] 兼容性验证：确保 eval.py 和测试不受影响
- **状态：** pending

### 阶段 4：验证与测试
- [ ] 运行 `pytest` 确保所有测试通过
- [ ] 用小数据集验证训练时间变化
- [ ] 确认 GPU 利用率提升
- **状态：** pending

### 阶段 5：交付
- [ ] 更新 AGENTS.md 中的训练命令（添加 --num-workers 默认值建议）
- [ ] 提交 commit
- **状态：** pending

## 关键问题
1. ~~num_workers 最优值是多少？~~ → 建议 4，可通过 `--num-workers` 参数调整
2. 批量 HDF5 读取与 PyTorch DataLoader 的兼容性？→ 通过自定义 collate_fn 实现
3. 是否需要重建 HDF5 数据集？→ 不需要，仅改代码

## 已做决策
| 决策 | 理由 |
|------|------|
| num_workers 默认从 0 改为 4 | 单进程加载导致 GPU 大量空闲，多进程可并行加载 |
| 新增 `pin_memory=True` | 加速 CPU→GPU 数据传输（~20% 传输带宽提升） |
| 自定义 collate_fn 做批量 HDF5 读取 | 减少 64 次独立 seek → 1 次批量读取 |
| 保留 `num_workers=0` 作为可选项 | Windows 兼容性（Windows 上 spawn 多进程有开销） |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
|      |         |         |

## 备注
- 训练集 ~192K 帧，batch_size=64 → ~3000 batches/epoch
- val 集 ~64K 帧 → ~1000 batches/epoch
- 当前每个 `__getitem__` 调用进行 6+ 次 HDF5 seek（keypoints, amplitude, phase, phase_cos, action, sample...）
- 每 epoch 约 192K × 6 = 115 万次 HDF5 seek 操作
