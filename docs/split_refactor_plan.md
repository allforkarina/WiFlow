# 训练数据划分重构计划

> 涉及 4 个文件：`data/memmap_dataset.py`、`dataloader.py`、`train.py`、`scripts/build_memmap.py`

**Goal:** 将当前帧级随机划分改为 subject-level 7:2:1 固定划分，env1/env2 同时出现在 train/val/test，S21-S40 预留跨域测试。

---

## 最终划分

```
Normalization:  S01-S20
Train:          S01-S07, S11-S17      (14 subjects)
Val:            S08-S09, S18-S19      (4 subjects)
Test:           S10, S20              (2 subjects)
Cross-domain:   S21-S40               (reserved)
```

---

## 1. `scripts/build_memmap.py` — 归一化统计量

- `--train-subjects` 默认值从 `S01..S10` 改为 `S01..S20`
- `stats.json` 新增 `"normalization_subjects": list(train_subjects)` 字段

## 2. `dataloader.py` — 定义默认划分

新增常量：
```python
DEFAULT_TRAIN_SUBJECTS = ("S01","S02","S03","S04","S05","S06","S07","S11","S12","S13","S14","S15","S16","S17")
DEFAULT_VAL_SUBJECTS   = ("S08","S09","S18","S19")
DEFAULT_TEST_SUBJECTS  = ("S10","S20")
CROSS_DOMAIN_SUBJECTS  = ("S21","S22","S23","S24","S25","S26","S27","S28","S29","S30","S31","S32","S33","S34","S35","S36","S37","S38","S39","S40")
KNOWN_SUBJECTS         = ("S01","S02",...,"S20")  # S01-S20
DEFAULT_SPLIT_MODE     = "subject_env_7_2_1"
```

工厂函数修改：
- `create_memmap_data_loader()` 新增参数：`split_mode=DEFAULT_SPLIT_MODE`、`known_subjects=KNOWN_SUBJECTS`、`train_subjects=DEFAULT_TRAIN_SUBJECTS`、`val_subjects=DEFAULT_VAL_SUBJECTS`、`test_subjects=DEFAULT_TEST_SUBJECTS`
- 根据 `split` 选择对应 subject 列表传递给 `MemmapDataset`
- `create_memmap_data_loaders()` 同样新增这些参数，透传

## 3. `data/memmap_dataset.py` — 重写 `_build_split`

### 修改 `__init__`
- 删除 `envs`、`train_subjects`、`test_subjects`、`random_val_ratio` 参数
- 新增 `split_mode: str = "subject_env_7_2_1"`
- 新增 `known_subjects: Iterable[str] | None = None`
- 新增 `train_subjects: Iterable[str] | None = None`
- 新增 `val_subjects: Iterable[str] | None = None`
- 新增 `test_subjects: Iterable[str] | None = None`
- `split` 支持值：`train` / `val` / `test` / `all` / `cross_domain`
- `split="cross_domain"` 时使用 `known_subjects=None` 返回全量数据（含 S21-S40）

### 重写 `_build_split`
- `split_mode="subject_env_7_2_1"`：
  - 根据 `split` 直接选择 `train_subjects` / `val_subjects` / `test_subjects`
  - 返回对应 subject 的所有帧（不随机切）
  - `all` 返回 `known_subjects` 的所有帧
  - `cross_domain` 返回不在 `known_subjects` 中的全部帧
- `split_mode="frame_random"`：
  - 保留当前逻辑（帧级随机切分，用于极限性能评估）
  - 使用 `known_subjects` 限定候选范围

### `_build_split` 新签名
```python
def _build_split(
    self,
    split: str,
    split_mode: str,
    known_subjects: Iterable[str] | None,
    train_subjects: Iterable[str] | None,
    val_subjects: Iterable[str] | None,
    test_subjects: Iterable[str] | None,
    random_val_ratio: float,
    seed: int,
) -> np.ndarray:
```

## 4. `train.py` — 无需修改

`create_memmap_data_loaders()` 已使用默认 7:2:1 划分，`train.py` 无须显式指定 subject 列表。

## 5. `eval.py` — 无需修改

`split="test"` 已映射到 `test_subjects=S10,S20`，不需要改 eval 调用。

## 6. 参数传递链

```
dataloader.py (定义默认常量)
  → MemmapDataset.__init__ (接收并存储 subject 列表 + split_mode)
    → _build_split (根据 split_mode 和 split 选择数据)
```
