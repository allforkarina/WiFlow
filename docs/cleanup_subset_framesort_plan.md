# 删除 subset_size + 修正帧排序 修改计划

**Goal:** 删除训练子集功能，修正帧文件排序为数值排序。

---

## 1. `train.py` — 删除 subset_size 功能

- 删除 `Subset` import（`from torch.utils.data import DataLoader, Subset` → `from torch.utils.data import DataLoader`）
- 删除 `maybe_subset_loader()` 函数
- `TrainConfig` 删除 `subset_size: int | None = None`
- 删除 `--subset-size` 命令行参数
- `run_training()` 中直接使用 `loaders["train"]` 和 `loaders["val"]`，不再通过 `maybe_subset_loader()`

## 2. `scripts/build_memmap.py` — 修正帧排序

`process_trial()` 中的 `common = sorted(set(mat_stems) & set(npy_stems))` 当前使用字符串排序，导致 `frame10` 排在 `frame2` 前。

修改：
- 新增 `_parse_frame_number(stem: str) -> int` 辅助函数，从 `"frame123"` 提取 `123`
- stem 不符合 `frame<num>` 格式时抛出 `ValueError`
- `common` 改为按 `_parse_frame_number(stem)` 的数值排序
- `frame_idx` 改为使用提取出的数值

## 3. 验证

```bash
# 验证语法
python -c "import ast; ast.parse(open('train.py', encoding='utf-8').read()); ast.parse(open('scripts/build_memmap.py', encoding='utf-8').read()); print('Syntax OK')"

# 验证无残留
grep -rn "subset_size\|maybe_subset\|Subset" train.py  # 应无输出
```
