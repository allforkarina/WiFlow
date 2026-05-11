# 发现与决策 — 训练速度诊断

## 需求
定位训练速度慢的原因，制定优化方案。当前 ~480s/epoch（含 train + val）。

## 研究发现

### 1. 量化分析

| 指标 | 数值 |
|------|------|
| Train 帧数 | 27 actions × 4 envs × 6 samples × 297 frames = **192,456** |
| Val 帧数 | 27 × 4 × 2 × 297 = **64,152** |
| Batch size | 64 |
| Train batches/epoch | ~3,007 |
| Val batches/epoch | ~1,003 |
| 每 batch 耗时 | 480s / (3007+1003) ≈ **120ms** |

### 2. 时间分布估算（per batch, ~120ms）

| 环节 | 估计耗时 | 占比 |
|------|---------|------|
| HDF5 随机读取 + CPU 归一化 | 60-80ms | 50-65% |
| CPU→GPU 数据传输 | 3-5ms | 3-4% |
| GPU forward pass | 15-20ms | 13-17% |
| GPU backward pass + optimizer | 25-35ms | 21-29% |
| Metrics + 日志 | 2-3ms | 2% |

### 3. 根因分析

**主因：`num_workers=0`（默认值）导致数据加载和 GPU 计算完全串行**

```
当前流程（num_workers=0）：
[CPU 加载 batch] → [GPU forward] → [GPU backward] → [CPU 加载 batch] → ...
      60-80ms            20ms            30ms             60-80ms

GPU 利用率 ≈ (20+30) / 120 = 42%
```

**次因：每个 `__getitem__` 进行 6+ 次独立 HDF5 seek**

```python
# 每个样本的 I/O 操作：
h5_file["keypoints"][frame_index]      # seek 1
h5_file["csi_amplitude"][frame_index]  # seek 2
h5_file["csi_phase"][frame_index]      # seek 3
h5_file["csi_phase_cos"][frame_index]  # seek 4
h5_file["action"][frame_index]         # seek 5
h5_file["environment"][frame_index]    # seek 6
```

每 epoch = 192K 样本 × 6 次 seek = **115 万次 HDF5 seek**。即使 SSD 延迟低（~0.1ms），累计也达 115s。

**次因：shuffle=True 导致 HDF5 随机访问**

shuffle 后 batch 内 64 个样本的 HDF5 位置完全随机，无法利用 HDF5 chunk 缓存。

### 4. HDF5 chunk 缓存检查

HDF5 默认 chunk 大小通常为数据集维度的 1/10。以 `csi_amplitude` 为例，shape 为 (320760, 3, 114, 10)，默认 chunk 可能为 (1000, 3, 114, 10)。相邻 frame 大概率在同一 chunk 内，但 shuffle 后相邻 frame 几乎不可能在同一 chunk。

## 技术决策
| 决策 | 理由 |
|------|------|
| `num_workers` 默认改为 4 | 最大收益，改动最小 |
| 添加 `pin_memory=True` | 加速 CPU→GPU 传输 |
| 自定义 `collate_fn` 批量读取 HDF5 | 减少 seek 次数从 384 次→6 次 per batch |
| 验证集每 N epoch 运行一次 | 减少 val 开销 |
| checkpoint 仅在 best 时保存（不含 last.pth） | 减少 I/O |

## 资源
- `train.py:49` — num_workers 默认值定义
- `train.py:220-266` — run_epoch 训练循环
- `dataloader.py:764-806` — MMFiPoseDataset.__getitem__
- `dataloader.py:810-843` — create_data_loader
- `dataloader.py:846-869` — create_data_loaders

## 视觉/浏览器发现
无

---
*每执行2次查看/浏览器/搜索操作后更新此文件*
*防止视觉信息丢失*
