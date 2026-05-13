# WiFlow Models 模块架构分析文档

> 生成日期: 2026-05-14 | 项目版本: NPY Memmap 后端 + OpenPose18

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [数据流全景图](#2-数据流全景图)
3. [活跃模块详解](#3-活跃模块详解)
   - [3.1 skeleton.py — 骨架拓扑定义](#31-skeletonpy--骨架拓扑定义)
   - [3.2 wiflow_spatial_encoder.py — CSI 空间编码器](#32-wiflow_spatial_encoderpy--csi-空间编码器)
   - [3.3 wiflow_axial_encoder.py — 轴向注意力编码器](#33-wiflow_axial_encoderpy--轴向注意力编码器)
   - [3.4 wiflow_model.py — 主模型编排器](#34-wiflow_modelpy--主模型编排器)
   - [3.5 wiflow_joint_decoder.py — 联合交叉注意力解码器](#35-wiflow_joint_decoderpy--联合交叉注意力解码器)
   - [3.6 wiflow_hierarchical_joint_decoder.py — 层级化关节解码器](#36-wiflow_hierarchical_joint_decoderpy--层级化关节解码器)
   - [3.7 wiflow_heatmap_decoder.py — MSFN 热图解码器](#37-wiflow_heatmap_decoderpy--msfn-热图解码器)
4. [遗留模块详解](#4-遗留模块详解)
   - [4.1 wiflow_attention_pooler.py — 注意力池化器](#41-wiflow_attention_poolerpy--注意力池化器)
   - [4.2 wiflow_skeleton_decoder.py — 骨架感知解码器](#42-wiflow_skeleton_decoderpy--骨架感知解码器)
   - [4.3 wiflow_spatial_temporal_fuser.py — 时空融合器](#43-wiflow_spatial_temporal_fuserpy--时空融合器)
   - [4.4 wiflow_temporal_encoder.py — 时序编码器](#44-wiflow_temporal_encoderpy--时序编码器)
5. [模块依赖关系图](#5-模块依赖关系图)
6. [删减与合并建议](#6-删减与合并建议)

---

## 1. 总体架构概览

`models/` 目录包含 11 个 Python 模块，实现 WiFlow 端到端姿态估计模型。按当前使用状态分为两类：

| 类别 | 模块数 | 说明 |
|------|--------|------|
| **活跃模块** | 7 | 当前 forward 路径直接使用 |
| **遗留模块** | 4 | 历史架构残留，当前 forward 路径未使用 |

### 活跃模块清单

| 文件 | 核心类 | 功能定位 |
|------|--------|----------|
| `skeleton.py` | — (常量/工具函数) | OpenPose18 骨架拓扑定义 |
| `wiflow_spatial_encoder.py` | `WiFlowSpatialEncoder`, `SymmetricResidualDownsampleBlock` | CSI 空间特征提取与下采样 |
| `wiflow_axial_encoder.py` | `WiFlowAxialEncoder` | 轴向注意力（空间+时间）编码 |
| `wiflow_model.py` | `WiFlowModel` | 端到端模型编排 |
| `wiflow_joint_decoder.py` | `WiFlowJointDecoder`, `WiFlowJointCrossAttentionLayer` | 联合查询交叉注意力解码（默认） |
| `wiflow_hierarchical_joint_decoder.py` | `WiFlowHierarchicalJointDecoder`, `WiFlowHierarchicalJointDecoderStage` | 层级化粗到细关节解码 |
| `wiflow_heatmap_decoder.py` | `WiFlowMSFNDecoder`, `WiFlowHeatmapDecoder`, `WiFlowPAPM` | 多阶段 PCM/PAF 热图解码 |

### 遗留模块清单

| 文件 | 核心类 | 功能定位 | 遗留原因 |
|------|--------|----------|----------|
| `wiflow_attention_pooler.py` | `WiFlowAttentionPooler` | 全局注意力池化 | 被 joint decoder 的 cross-attention 替代 |
| `wiflow_skeleton_decoder.py` | `WiFlowSkeletonDecoder` | 骨架感知解码 | 被 joint decoder 替代 |
| `wiflow_spatial_temporal_fuser.py` | `WiFlowSpatialTemporalFuser` | 多帧时空融合 | 序列模式已移除 |
| `wiflow_temporal_encoder.py` | `WiFlowTemporalEncoder` | 时序自注意力 | 序列模式已移除 |

---

## 2. 数据流全景图

```
CSI Input [B, 3, 114, 64]
    │
    ▼
WiFlowSpatialEncoder
    │  antenna_mixer → feature_stem → resblock1(stride=2) → resblock2(stride=2) → resblock3(stride=1)
    │  子载波: 114 → 57 → 29         时间: 64 → 32 → 16
    ▼
Feature Map [B, 128, 29, 16]
    │
    ▼
WiFlowAxialEncoder (mode: spatial_then_temporal | temporal_then_spatial | parallel_sum | parallel_concat)
    │  spatial_attention → temporal_attention → channel_projection (128 → 256)
    ▼
Feature Map [B, 256, 29, 16]
    │
    ├── decoder_type = "joint" ──────────► WiFlowJointDecoder ──────────► [B, 18, 2]
    ├── decoder_type = "hierarchical" ───► WiFlowHierarchicalJointDecoder ► [B, 18, 2]
    └── decoder_type = "heatmap_msfn" ───► WiFlowMSFNDecoder ────────────► {keypoints: [B,18,2], stages: [...]}
```

---

## 3. 活跃模块详解

### 3.1 skeleton.py — 骨架拓扑定义

**文件路径**: `models/skeleton.py`

**状态**: ✅ 活跃（被所有 decoder 依赖）

**核心功能**:
定义 OpenPose18 人体关键点骨架拓扑，包括 18 个关键点、19 条骨骼连接边，以及归一化邻接矩阵构建函数。

**关键常量**:

| 常量 | 值 | 说明 |
|------|-----|------|
| `NUM_OPENPOSE_KEYPOINTS` | `18` | 关键点总数 |
| `OPENPOSE_BONE_EDGES` | `((0,1), (1,2), ..., (5,11))` | 19 条骨骼边 |

**OpenPose18 关键点索引**:
```
 0: Nose        1: Neck         2: RShoulder    3: RElbow      4: RWrist
 5: LShoulder   6: LElbow       7: LWrist       8: RHip        9: RKnee
10: RAnkle     11: LHip        12: LKnee       13: LAnkle     14: REye
15: LEye       16: REar        17: LEar
```

**骨骼边（19 条）**:
```
(0,1) Nose→Neck        (1,2) Neck→RShoulder    (2,3) RShoulder→RElbow
(3,4) RElbow→RWrist    (1,5) Neck→LShoulder    (5,6) LShoulder→LElbow
(6,7) LElbow→LWrist    (1,8) Neck→RHip         (8,9) RHip→RKnee
(9,10) RKnee→RAnkle    (1,11) Neck→LHip        (11,12) LHip→LKnee
(12,13) LKnee→LAnkle   (0,14) Nose→REye        (14,16) REye→REar
(0,15) Nose→LEye       (15,17) LEye→LEar       (2,8) RShoulder→RHip
(5,11) LShoulder→LHip
```

**核心函数**:

| 函数 | 签名 | 功能 |
|------|------|------|
| `build_normalized_adjacency` | `(num_nodes=18, edges=OPENPOSE_BONE_EDGES) -> Tensor[18,18]` | 构建对称归一化邻接矩阵（含自环），使用 D^{-1/2} A D^{-1/2} 归一化 |

**依赖关系**:
- 无内部依赖（纯工具模块）
- 被 `wiflow_joint_decoder.py`, `wiflow_hierarchical_joint_decoder.py`, `wiflow_skeleton_decoder.py`, `wiflow_heatmap_decoder.py` 引用

---

### 3.2 wiflow_spatial_encoder.py — CSI 空间编码器

**文件路径**: `models/wiflow_spatial_encoder.py`

**状态**: ✅ 活跃（forward 路径第一步）

**核心功能**:
将原始 CSI 幅度信号 `[B, 3, 114, 64]` 编码为空间特征图 `[B, 128, 29, 16]`，通过天线混合、特征提取和对称时空下采样实现。

**关键算法**:

1. **天线混合 (Antenna Mixing)**: 1×1 卷积跨天线通道交互，保持物理维度语义
2. **特征茎 (Feature Stem)**: 3×5 非对称卷积，在子载波方向使用更大核（5）捕获频率模式
3. **对称残差下采样块 (SymmetricResidualDownsampleBlock)**: 核心创新——时间维度和子载波维度**同步**下采样

**数据流**:

```
Input:  [B, 3, 114, 64]
  │ _to_conv_layout: permute(0,1,3,2)
  ▼
[B, 3, 64, 114]
  │ antenna_mixer: Conv2d(3→3, 1×1) + BN + ReLU
  ▼
[B, 3, 64, 114]
  │ feature_stem: Conv2d(3→32, 3×5, pad=1,2) + BN + ReLU
  ▼
[B, 32, 64, 114]
  │ resblock1: SymmetricResidualDownsampleBlock(32→64, stride=2)
  │   时间: 64→32, 子载波: 114→57
  ▼
[B, 64, 32, 57]
  │ resblock2: SymmetricResidualDownsampleBlock(64→128, stride=2)
  │   时间: 32→16, 子载波: 57→29
  ▼
[B, 128, 16, 29]
  │ resblock3: SymmetricResidualDownsampleBlock(128→128, stride=1)
  │   保持分辨率不变
  ▼
[B, 128, 16, 29]
  │ _to_model_layout: transpose(2,3)
  ▼
Output: [B, 128, 29, 16]
```

**类与接口**:

| 类 | 参数 | 说明 |
|-----|------|------|
| `SymmetricResidualDownsampleBlock` | `in_channels, out_channels, stride` | 对称下采样残差块，主路径为两个 3×3 卷积，shortcut 为 1×1 卷积 |
| `WiFlowSpatialEncoder` | `input_channels=3` | 空间编码器主类，仅接受 3 通道输入 |

**关键设计决策**:
- 输入通道严格限制为 3（仅 CSI 幅度），不支持多特征组
- 使用 `_to_conv_layout` / `_to_model_layout` 在通道优先和卷积优先布局间转换
- 总下采样倍率：子载波 114→29 (≈4×)，时间 64→16 (4×)

**依赖关系**:
- 仅依赖 `torch.nn`
- 被 `wiflow_model.py` 引用

---

### 3.3 wiflow_axial_encoder.py — 轴向注意力编码器

**文件路径**: `models/wiflow_axial_encoder.py`

**状态**: ✅ 活跃（forward 路径第二步）

**核心功能**:
对空间编码器输出的特征图 `[B, 128, 29, 16]` 施加轴向注意力（沿空间轴和时间轴分别做 self-attention），输出 `[B, 256, 29, 16]`。

**关键算法**:

1. **空间注意力**: 将 `[B, 128, 29, T]` reshape 为 `[B*T, 29, 128]`，沿 29 个空间位置做 self-attention
2. **时间注意力**: 将 `[B, 128, 29, T]` reshape 为 `[B*29, T, 128]`，沿 T 个时间步做 self-attention
3. **通道投影**: 1×1 卷积将 128 通道升维到 256

**四种模式**:

| 模式 | 常量 | 计算流程 |
|------|------|----------|
| `spatial_then_temporal` | 默认 | spatial_attn → temporal_attn → proj |
| `temporal_then_spatial` | — | temporal_attn → spatial_attn → proj |
| `parallel_sum` | — | spatial_attn ∥ temporal_attn → sum → proj |
| `parallel_concat` | — | spatial_attn ∥ temporal_attn → concat → proj |

**数据流（以 spatial_then_temporal 为例）**:

```
Input:  [B, 128, 29, 16]
  │ _prepare_spatial_attention_input: permute(0,3,2,1) → reshape(B*16, 29, 128)
  ▼
[B*16, 29, 128]
  │ spatial_attention (MultiheadAttention, 8 heads) + residual + LayerNorm
  ▼
[B*16, 29, 128]
  │ _restore_spatial_attention_output: reshape → permute
  ▼
[B, 128, 29, 16]
  │ _prepare_temporal_attention_input: permute(0,2,3,1) → reshape(B*29, 16, 128)
  ▼
[B*29, 16, 128]
  │ temporal_attention (MultiheadAttention, 8 heads) + residual + LayerNorm
  ▼
[B*29, 16, 128]
  │ _restore_temporal_attention_output: reshape → permute
  ▼
[B, 128, 29, 16]
  │ channel_projection: Conv2d(128→256, 1×1)
  ▼
Output: [B, 256, 29, 16]
```

**类与接口**:

| 类 | 参数 | 说明 |
|-----|------|------|
| `WiFlowAxialEncoder` | `mode="spatial_then_temporal"` | 轴向注意力编码器，支持 4 种模式 |

**导出常量**:
- `AXIAL_ENCODER_MODES`: `("spatial_then_temporal", "temporal_then_spatial", "parallel_sum", "parallel_concat")`

**依赖关系**:
- 仅依赖 `torch.nn`
- 被 `wiflow_model.py` 引用

---

### 3.4 wiflow_model.py — 主模型编排器

**文件路径**: `models/wiflow_model.py`

**状态**: ✅ 活跃（顶层入口）

**核心功能**:
端到端 WiFlow 模型编排器，串联 SpatialEncoder → AxialEncoder → Decoder，将 CSI 特征映射为 OpenPose18 坐标。

**数据流**:

```
Input: [B, 3, 114, 64]
  │
  ├── spatial_encoder (WiFlowSpatialEncoder)
  │   └── [B, 128, 29, 16]
  ├── axial_encoder (WiFlowAxialEncoder)
  │   └── [B, 256, 29, 16]
  └── decoder (按 decoder_type 选择)
      ├── "joint"           → WiFlowJointDecoder           → [B, 18, 2]
      ├── "hierarchical"    → WiFlowHierarchicalJointDecoder → [B, 18, 2]
      └── "heatmap_msfn"    → WiFlowMSFNDecoder
                                → decode_pcm_argmax(stages[-1]["pcm"])
                                → {"keypoints": [B,18,2], "stages": [...]}
```

**类与接口**:

| 类 | 参数 | 说明 |
|-----|------|------|
| `WiFlowModel` | `input_channels=3, axial_mode="spatial_then_temporal", decoder_type="joint", heatmap_size=36` | 端到端模型 |

**关键方法**:

| 方法 | 功能 |
|------|------|
| `forward(x)` | 主前向传播，输入 `[B,3,114,64]`，输出坐标或热图 |
| `decode_features(x)` | 解码器输出后处理，heatmap_msfn 模式下额外执行 argmax 解码 |

**导出常量**:
- `DECODER_TYPES`: `("joint", "hierarchical", "heatmap_msfn")`

**输入约束**:
- 严格 4D 输入 `[B, 3, 114, 64]`，5D 输入会抛出 `ValueError`
- 不再支持 `sequence_length` 参数和序列模式

**依赖关系**:
- 依赖 `wiflow_spatial_encoder`, `wiflow_axial_encoder`, `wiflow_joint_decoder`, `wiflow_hierarchical_joint_decoder`, `wiflow_heatmap_decoder`
- 依赖外部 `pose_targets.decode_pcm_argmax`
- 被 `train.py`, `eval.py` 引用

---

### 3.5 wiflow_joint_decoder.py — 联合交叉注意力解码器

**文件路径**: `models/wiflow_joint_decoder.py`

**状态**: ✅ 活跃（默认 decoder）

**核心功能**:
使用 18 个可学习 joint query 向量，通过多层交叉注意力从空间特征图中检索关节点坐标，并结合 GNN 骨架约束和自注意力精炼。

**关键算法**:

1. **Joint Query 机制**: 18 个可学习嵌入向量 `[18, 256]`，每个对应一个关节点
2. **交叉注意力**: joint queries 作为 Q，空间 tokens `[B, 464, 256]`（29×16 flatten）作为 K/V
3. **GNN 骨架约束**: 使用归一化邻接矩阵 `[18, 18]` 在关节间传播信息
4. **自注意力精炼**: 关节间 self-attention 建模全局关节依赖
5. **坐标回归头**: Linear(256→128) → SiLU → Linear(128→2)

**数据流**:

```
Input: [B, 256, 29, 16]
  │ flatten_tokens: flatten(2).transpose(1,2)
  ▼
Spatial Tokens: [B, 464, 256]
  │
  ├── Joint Queries: [18, 256] → expand → [B, 18, 256]
  │
  ├── Cross-Attention Layer × 3:
  │     joint_queries (Q) × spatial_tokens (K,V)
  │     → residual + LayerNorm
  │     → FeedForward (256→512→256) + residual + LayerNorm
  │
  ├── GNN: adjacency @ projection(h) → SiLU → residual + LayerNorm
  │
  ├── Joint Self-Attention: h (Q,K,V) → residual + LayerNorm
  │
  └── Coordinate Head: Linear(256→128) → SiLU → Linear(128→2)
  ▼
Output: [B, 18, 2]
```

**类与接口**:

| 类 | 参数 | 说明 |
|-----|------|------|
| `WiFlowJointCrossAttentionLayer` | `embedding_dim=256` | 单层交叉注意力 + FFN |
| `WiFlowJointDecoder` | `num_layers=3` | 联合解码器主类 |

**关键设计**:
- 默认 3 层交叉注意力，每层 8 头
- GNN 使用预计算的归一化邻接矩阵（`build_normalized_adjacency`）
- 坐标头输出未归一化的 `[x, y]` 坐标

**依赖关系**:
- 依赖 `skeleton.py`（`NUM_OPENPOSE_KEYPOINTS`, `build_normalized_adjacency`）
- 被 `wiflow_model.py` 引用

---

### 3.6 wiflow_hierarchical_joint_decoder.py — 层级化关节解码器

**文件路径**: `models/wiflow_hierarchical_joint_decoder.py`

**状态**: ✅ 活跃（ablation decoder）

**核心功能**:
分阶段从粗到细解码关节点坐标。先解码躯干核心关节（neck, shoulders, hips），再利用已解码关节的上下文信息解码四肢关节，最后解码面部关键点。

**关键算法**:

1. **三阶段解码**:
   - Stage 1（无上下文）: 躯干关节 `(0,1,2,5,8,11)` — Nose, Neck, R/LShoulder, R/LHip
   - Stage 2（有上下文）: 四肢关节 `(3,4,6,7,9,10,12,13)` — Elbows, Wrists, Knees, Ankles
   - Stage 3（有上下文）: 面部关键点 `(14,15,16,17)` — Eyes, Ears

2. **上下文传递**: 后续 stage 通过 cross-attention 关注前面 stage 已解码的关节嵌入

3. **重排序**: 按 stage 顺序解码后，通过 `openpose_order` 索引重排回标准 OpenPose18 顺序

**数据流**:

```
Input: [B, 256, 29, 16]
  │ flatten_tokens
  ▼
Spatial Tokens: [B, 464, 256]
  │
  ├── Stage 1 (躯干, 6 joints, no context):
  │     query[躯干索引] × spatial_tokens → spatial_attn → FFN
  │
  ├── Stage 2 (四肢, 8 joints, context=stage1_output):
  │     query[四肢索引] × spatial_tokens → spatial_attn
  │     → context_attn(stage1_output) → FFN
  │
  ├── Stage 3 (面部, 4 joints, context=stage1+stage2):
  │     query[面部索引] × spatial_tokens → spatial_attn
  │     → context_attn(stage1+stage2) → FFN
  │
  ├── 重排序: concat → [:, openpose_order]
  ├── GNN + Joint Self-Attention
  └── Coordinate Head
  ▼
Output: [B, 18, 2]
```

**类与接口**:

| 类 | 参数 | 说明 |
|-----|------|------|
| `WiFlowHierarchicalJointDecoderStage` | `has_context, embedding_dim=256` | 单阶段解码块，可选上下文注意力 |
| `WiFlowHierarchicalJointDecoder` | — | 层级化解码器主类 |

**依赖关系**:
- 依赖 `skeleton.py`
- 被 `wiflow_model.py` 引用

---

### 3.7 wiflow_heatmap_decoder.py — MSFN 热图解码器

**文件路径**: `models/wiflow_heatmap_decoder.py`

**状态**: ✅ 活跃（ablation decoder）

**核心功能**:
Multi-Stage Feature Network (MSFN) 风格的多阶段 PCM/PAF 热图解码器，通过 PAPM（Pose-Aware Feature Modulation）反馈机制逐步精炼热图预测。

**关键算法**:

1. **WiFlowHeatmapDecoder（单阶段）**: 4 层 3×3 卷积共享骨干 → 1×1 bottleneck → PCM head + PAF head
2. **WiFlowPAPM（姿态感知调制）**: 利用上一阶段 PCM/PAF 热图，通过通道门控（全局平均/最大池化 → MLP → Sigmoid）和空间门控（卷积 → Sigmoid）调制特征
3. **WiFlowMSFNDecoder（多阶段编排）**: 输入投影 → 上采样到 heatmap_size → 迭代 decoder → PAPM 反馈

**数据流**:

```
Input: [B, 256, 29, 16]
  │ input_projection: Conv2d(256→128, 1×1) + BN + SiLU
  │ interpolate → [B, 128, H, H]  (H = heatmap_size, default 36)
  ▼
[B, 128, 36, 36]
  │
  ├── Stage 1: WiFlowHeatmapDecoder
  │     shared(4×Conv3×3) → bottleneck → PCM:[B,18,36,36] + PAF:[B,38,36,36]
  │
  ├── PAPM 1: channel_gate + spatial_gate → refine
  │
  ├── Stage 2: WiFlowHeatmapDecoder
  │     → PCM:[B,18,36,36] + PAF:[B,38,36,36]
  │
  ├── PAPM 2: ...
  │
  └── Stage 3: WiFlowHeatmapDecoder
        → PCM:[B,18,36,36] + PAF:[B,38,36,36]
  ▼
Output: [
  {"pcm": [B,18,36,36], "paf": [B,38,36,36]},  # stage 1
  {"pcm": [B,18,36,36], "paf": [B,38,36,36]},  # stage 2
  {"pcm": [B,18,36,36], "paf": [B,38,36,36]},  # stage 3
]
```

**通道数说明**:
- PCM: 18 通道（OpenPose18 每个关键点一个热图）
- PAF: 38 通道（19 条骨骼边 × 2 方向分量）

**类与接口**:

| 类 | 参数 | 说明 |
|-----|------|------|
| `WiFlowHeatmapDecoder` | `feature_channels=128, hidden_channels=512, pcm_channels=18, paf_channels=38` | 单阶段热图解码器 |
| `WiFlowPAPM` | `feature_channels=128, heatmap_channels=56` | 姿态感知特征调制模块 |
| `WiFlowMSFNDecoder` | `input_channels=256, feature_channels=128, hidden_channels=512, stages=3, heatmap_size=36` | 多阶段编排器 |

**依赖关系**:
- 依赖 `skeleton.py`（`NUM_OPENPOSE_KEYPOINTS`, `OPENPOSE_BONE_EDGES`）
- 被 `wiflow_model.py` 引用
- 在 `WiFlowModel.decode_features` 中通过 `decode_pcm_argmax` 将最后一阶段 PCM 解码为坐标

---

## 4. 遗留模块详解

### 4.1 wiflow_attention_pooler.py — 注意力池化器

**文件路径**: `models/wiflow_attention_pooler.py`

**状态**: ❌ 遗留（未被当前 forward 路径使用）

**核心功能**:
使用一个可学习的全局 query token，通过交叉注意力将 `[B, 256, 29, 10]` 空间特征图池化为单个 `[B, 256]` 全局嵌入向量。

**数据流**:
```
Input: [B, 256, 29, 10]
  │ flatten_tokens: permute(0,2,3,1) → reshape(B, 290, 256)
  ▼
[B, 290, 256]
  │ global_query [1,1,256] × tokens → cross_attention → residual + LayerNorm → squeeze
  ▼
Output: [B, 256]
```

**遗留原因**: 当前架构使用 joint query 交叉注意力直接从空间 tokens 解码坐标，不再需要先池化为全局向量。

**依赖关系**: 仅依赖 `torch.nn`，在 `__init__.py` 中仍被导出。

---

### 4.2 wiflow_skeleton_decoder.py — 骨架感知解码器

**文件路径**: `models/wiflow_skeleton_decoder.py`

**状态**: ❌ 遗留（未被当前 forward 路径使用）

**核心功能**:
从全局池化向量 `[B, 256]` 出发，通过 joint queries + GNN + self-attention 解码 18 个关节点坐标。

**数据流**:
```
Input: [B, 256]
  │ joint_queries [18,256] + input.unsqueeze(1) → LayerNorm
  ▼
[B, 18, 256]
  │ GNN: adjacency @ projection → SiLU → residual + LayerNorm
  │ Joint Self-Attention → residual + LayerNorm
  │ Coordinate Head: Linear(256→128) → SiLU → Linear(128→2)
  ▼
Output: [B, 18, 2]
```

**与 WiFlowJointDecoder 的区别**:
- SkeletonDecoder 输入是**已池化的全局向量** `[B, 256]`，不直接访问空间 tokens
- JointDecoder 输入是**完整空间特征图** `[B, 256, 29, 16]`，通过交叉注意力检索

**遗留原因**: 被 WiFlowJointDecoder 替代，后者直接从空间特征图检索信息，信息损失更小。

**依赖关系**: 依赖 `skeleton.py`，在 `__init__.py` 中仍被导出。

---

### 4.3 wiflow_spatial_temporal_fuser.py — 时空融合器

**文件路径**: `models/wiflow_spatial_temporal_fuser.py`

**状态**: ❌ 遗留（序列模式已移除）

**核心功能**:
将多帧空间特征图 `[B, N, 256, 29, 10]` 通过时序 self-attention 融合为单帧 `[B, 256, 29, 10]`，取中间帧作为输出。

**数据流**:
```
Input: [B, N, 256, 29, 10]
  │ permute(0,3,4,1,2) → reshape(B*29*10, N, 256)
  ▼
[B*290, N, 256]
  │ + position_embedding [1, N, 256]
  │ self_attention → residual + LayerNorm
  │ 取 middle_index (N//2)
  ▼
[B*290, 256]
  │ reshape → permute
  ▼
Output: [B, 256, 29, 10]
```

**遗留原因**: 当前架构已移除 `sequence_length` 参数和序列模式，CSI 的 64 个时间步在单样本内处理，不再需要跨帧融合。

**依赖关系**: 仅依赖 `torch.nn`，在 `__init__.py` 中仍被导出。

---

### 4.4 wiflow_temporal_encoder.py — 时序编码器

**文件路径**: `models/wiflow_temporal_encoder.py`

**状态**: ❌ 遗留（序列模式已移除）

**核心功能**:
对时序 token 序列 `[B, N, 256]` 施加 self-attention，取中间 token 作为输出。

**数据流**:
```
Input: [B, N, 256]
  │ + position_embedding [1, N, 256]
  │ self_attention → residual + LayerNorm
  │ 取 middle_index (N//2)
  ▼
Output: [B, 256]
```

**遗留原因**: 与 SpatialTemporalFuser 相同，序列模式已移除。

**依赖关系**: 仅依赖 `torch.nn`，在 `__init__.py` 中仍被导出。

---

## 5. 模块依赖关系图

```
skeleton.py ◄────────────────────────────────────────────┐
    │                                                     │
    ├──────────────────────────────────────┐              │
    │                                      │              │
    ▼                                      ▼              │
wiflow_joint_decoder.py          wiflow_hierarchical      │
    │                              _joint_decoder.py       │
    │                                      │              │
    │                                      │              │
    │  ┌───────────────────────────────────┘              │
    │  │                                                  │
    ▼  ▼              wiflow_heatmap_decoder.py ◄─────────┘
wiflow_model.py ◄──────────────┘
    │
    ├── wiflow_spatial_encoder.py
    └── wiflow_axial_encoder.py

遗留模块（无活跃依赖）:
    wiflow_attention_pooler.py
    wiflow_skeleton_decoder.py ◄── skeleton.py
    wiflow_spatial_temporal_fuser.py
    wiflow_temporal_encoder.py
```

---

## 6. 删减与合并建议

### 6.1 可安全删除的模块

| 模块 | 理由 | 影响范围 |
|------|------|----------|
| `wiflow_attention_pooler.py` | 功能已被 joint decoder 的 cross-attention 完全替代 | 需从 `__init__.py` 移除导出 |
| `wiflow_skeleton_decoder.py` | 功能已被 joint decoder 替代，且信息损失更大 | 需从 `__init__.py` 移除导出 |
| `wiflow_spatial_temporal_fuser.py` | 序列模式已移除，无使用场景 | 需从 `__init__.py` 移除导出 |
| `wiflow_temporal_encoder.py` | 序列模式已移除，无使用场景 | 需从 `__init__.py` 移除导出 |

### 6.2 建议保留的模块

| 模块 | 保留理由 |
|------|----------|
| `skeleton.py` | 被所有 decoder 依赖，是骨架拓扑的单一事实来源 |
| `wiflow_spatial_encoder.py` | 核心编码器，包含对称下采样创新设计 |
| `wiflow_axial_encoder.py` | 核心编码器，支持 4 种轴向注意力模式 |
| `wiflow_model.py` | 顶层编排器，train/eval 的入口 |
| `wiflow_joint_decoder.py` | 默认 decoder，当前最佳方案 |
| `wiflow_hierarchical_joint_decoder.py` | ablation decoder，用于对比实验 |
| `wiflow_heatmap_decoder.py` | ablation decoder，MSFN 风格热图方案 |

### 6.3 合并建议

当前模块划分已经较为合理，不建议进一步合并：
- `WiFlowHeatmapDecoder` + `WiFlowPAPM` + `WiFlowMSFNDecoder` 三者紧密耦合，放在同一文件合理
- `WiFlowJointCrossAttentionLayer` 是 `WiFlowJointDecoder` 的内部组件，放在同一文件合理
- `SymmetricResidualDownsampleBlock` 是 `WiFlowSpatialEncoder` 的专用组件，放在同一文件合理

### 6.4 删除后的 `__init__.py` 预期

```python
from .skeleton import NUM_OPENPOSE_KEYPOINTS, OPENPOSE_BONE_EDGES, build_normalized_adjacency
from .wiflow_axial_encoder import AXIAL_ENCODER_MODES, WiFlowAxialEncoder
from .wiflow_heatmap_decoder import WiFlowHeatmapDecoder, WiFlowMSFNDecoder, WiFlowPAPM
from .wiflow_hierarchical_joint_decoder import WiFlowHierarchicalJointDecoder
from .wiflow_joint_decoder import WiFlowJointDecoder
from .wiflow_model import DECODER_TYPES, WiFlowModel
from .wiflow_spatial_encoder import WiFlowSpatialEncoder

__all__ = [
    "WiFlowModel",
    "WiFlowSpatialEncoder",
    "WiFlowAxialEncoder",
    "AXIAL_ENCODER_MODES",
    "DECODER_TYPES",
    "WiFlowJointDecoder",
    "WiFlowHierarchicalJointDecoder",
    "WiFlowHeatmapDecoder",
    "WiFlowMSFNDecoder",
    "WiFlowPAPM",
    "OPENPOSE_BONE_EDGES",
    "NUM_OPENPOSE_KEYPOINTS",
    "build_normalized_adjacency",
]
```