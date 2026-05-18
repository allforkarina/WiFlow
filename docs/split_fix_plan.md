# split_refactor 修复计划

三个问题的根因相同：`dataloader.py` 在外部预选了 `subjects` 只传给 `known_subjects`，没有传 `train_subjects`/`val_subjects`/`test_subjects`。加上 `cross_domain` 用 `None` 表示不过滤。`main()` 预览绕过工厂函数。

## 修复原则

- `dataloader.py` 始终传入全部 5 个 subject 列表
- `MemmapDataset._build_split` 删除 `None`=全量语义
- `main()` 走工厂函数

---

## 1. `dataloader.py` — 删除中间的 split→subjects 转换

### `create_memmap_data_loader`

删除整个 `if split_mode == "subject_env_7_2_1"` 分支和 `else` 分支。统一为：

```python
def create_memmap_data_loader(
    data_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    seed: int = 42,
    load_reference: bool = False,
    split_mode: str = DEFAULT_SPLIT_MODE,
    known_subjects: tuple[str, ...] = KNOWN_SUBJECTS,
    train_subjects: tuple[str, ...] = DEFAULT_TRAIN_SUBJECTS,
    val_subjects: tuple[str, ...] = DEFAULT_VAL_SUBJECTS,
    test_subjects: tuple[str, ...] = DEFAULT_TEST_SUBJECTS,
    cross_domain_subjects: tuple[str, ...] = CROSS_DOMAIN_SUBJECTS,
) -> DataLoader:
    dataset = MemmapDataset(
        data_dir=data_dir,
        split=split,
        split_mode=split_mode,
        known_subjects=known_subjects,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        cross_domain_subjects=cross_domain_subjects,
        seed=seed,
        load_reference=load_reference,
    )
    ...
```

### `create_memmap_data_loaders`

同样新增 `cross_domain_subjects` 参数并透传。

### `main()` preview

改为通过 `create_memmap_data_loader(batch_size=1, ...)` 获取 loader 后取 batch，不再直接 `MemmapDataset(...)`：

```python
for split in SPLIT_NAMES + ("cross_domain",):
    loader = create_memmap_data_loader(
        data_dir=data_dir, split=split, batch_size=1, num_workers=0, load_reference=True,
    )
    batch = next(iter(loader))
    print(f"{split}: {len(loader.dataset)} samples, subjects={set(batch['sample'])}")
```

## 2. `data/memmap_dataset.py` — 新增 `cross_domain_subjects`，删除 `None`=全量

### `__init__` 签名

新增参数：
```python
cross_domain_subjects: Iterable[str] | None = None,
```

存储：
```python
self.cross_domain_subjects = list(cross_domain_subjects) if cross_domain_subjects else []
```

### `_build_split` — `subject_env_7_2_1` 分支

```python
if split_mode == "subject_env_7_2_1":
    if split == "all":
        subjects = known_subjects or []
    elif split == "cross_domain":
        subjects = cross_domain_subjects or []
    elif split == "train":
        subjects = train_subjects or []
    elif split == "val":
        subjects = val_subjects or []
    elif split == "test":
        subjects = test_subjects or []
    else:
        raise ValueError(f"Unknown split: {split}")

    indices = [i for i in range(len(self._samples)) if sample_list[i] in set(subjects)]
    return np.asarray(sorted(indices), dtype=np.int64)
```

删除 `subjects is None` 的不过滤路径。空列表 `[]` 返回 0 个样本（安全失败）。

### `_build_split` — `frame_random` 分支

同样删除 `None`=全量语义。`known_subjects` 为 `None` 或空时返回空结果。

## 3. 额外约束

### 约束 1 & 2：`cross_domain_subjects` 必须贯穿全链路
- `__init__` 存储 → `self.cross_domain_subjects`
- `_build_split` 签名新增 `cross_domain_subjects`
- `self.indices = self._build_split(...)` 调用处传入

### 约束 3：preview 验证整个 dataset 而非单个 batch
```python
dataset = loader.dataset  # 直接检查 dataset
subjects = set()
for idx in dataset.indices:
    subjects.add(dataset._samples[idx])
print(f"{split}: {len(dataset)} samples, subjects={sorted(subjects)}")
```

### 约束 4：`frame_random` 空 `known_subjects`=空结果
接受此行为。以后统一从 `dataloader.py` 工厂函数创建 loader。

---

## 4. 验收

```bash
# 语法检查
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in [
    r'D:/Files/Projects/PythonProjects/PaperResuming/WiFlow/dataloader.py',
    r'D:/Files/Projects/PythonProjects/PaperResuming/WiFlow/data/memmap_dataset.py',
]]; print('Syntax OK')"
```
