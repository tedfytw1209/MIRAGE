# Public OCT B-scan CSV/Fold Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_cls_tuning_bscan.py` load the 4 public OCT B-scan datasets (duke14, glaucoma, oimhs, umn) from their real on-disk layout — raw per-slice volume folders plus a fold-split CSV — instead of the nonexistent pre-split `train/val/test/Class_x/` image folders it currently expects.

**Architecture:** A new `PublicOCTBscanDataset` (in `mutils/dataset_public_oct.py`, sibling to the existing `mutils/dataset_uf.py`) reads a fold CSV, filters by its `split` column, and resolves each row's middle slice to an image path via a per-dataset path-config dict added to `run_cls_tuning_bscan.py`. `process_args`/`build_dataset`/`get_output_dir` in that script are rewritten to use the CSV instead of `iterdir()`/`ImageFolder`. `run_cls_tuning_fundus.py` and its shell scripts are untouched.

**Tech Stack:** Python, pandas, PyTorch `Dataset`/`DataLoader`, PIL, bash (SLURM shell scripts).

**Note on testing:** This repo has no pytest/test suite (verified: no `tests/` directory, no `conftest.py`, no pytest config file anywhere in the repo). Verification steps below use small standalone Python scripts against synthetic data (in the scratchpad directory, not committed) instead of a pytest suite, since introducing a new test framework isn't part of this change's scope. Steps still follow "write the check, watch it fail, make it pass" — just via `python script.py` instead of `pytest`.

---

### Task 1: `PublicOCTBscanDataset` in `mutils/dataset_public_oct.py`

**Files:**
- Create: `mutils/dataset_public_oct.py`
- Verification script (not committed): `<scratchpad>/verify_dataset_public_oct.py`

- [ ] **Step 1: Write the verification script against synthetic data, and confirm it fails (module doesn't exist yet)**

Create `<scratchpad>/verify_dataset_public_oct.py` (use the scratchpad path from your environment, e.g. `C:\Users\teddy\AppData\Local\Temp\claude\f--github-MIRAGE\68308f3e-3c62-4166-8b5e-db9e5cc38151\scratchpad\verify_dataset_public_oct.py`):

```python
"""Synthetic smoke test for mutils.dataset_public_oct.PublicOCTBscanDataset.
Not part of the repo's test suite (there isn't one) -- run manually with
`python verify_dataset_public_oct.py` from the repo root (MIRAGE is on
sys.path via the cwd import, since mutils is a plain package there).
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, r'f:\github\MIRAGE')

from mutils.dataset_public_oct import PublicOCTBscanDataset


def make_synthetic_dataset(root: Path):
    # Two classes, two volumes each, 3 slices per volume.
    rows = []
    for folder, label in [('AMD', 0), ('NORMAL', 1)]:
        class_dir = root / folder
        class_dir.mkdir(parents=True, exist_ok=True)
        for patient_num in (1, 2):
            base = f'{folder}_{patient_num}_%02d'
            for slice_idx in (0, 1, 2):
                img_path = class_dir / (base % slice_idx + '.jpg')
                Image.new('RGB', (8, 8), color=(slice_idx * 10, 0, 0)).save(img_path)
            split = 'train' if patient_num == 1 else 'val'
            rows.append({
                'folder': folder, 'oct_imgname': base, 'slice_indices': '0-1-2',
                'label': label, 'split': split,
            })
    csv_path = root / 'fold0.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix='public_oct_test_'))
    try:
        csv_path = make_synthetic_dataset(tmp_root)

        train_ds = PublicOCTBscanDataset(csv_path, tmp_root, split='train')
        assert len(train_ds) == 2, f'expected 2 train samples, got {len(train_ds)}'
        image, label = train_ds[0]
        assert image.size == (8, 8), f'unexpected image size {image.size}'
        assert label in (0, 1), f'unexpected label {label}'

        val_ds = PublicOCTBscanDataset(csv_path, tmp_root, split='val')
        assert len(val_ds) == 2, f'expected 2 val samples, got {len(val_ds)}'

        test_ds = PublicOCTBscanDataset(csv_path, tmp_root, split='test')
        assert len(test_ds) == 0, f'expected 0 test samples, got {len(test_ds)}'

        # Middle slice of '0-1-2' is index 1 -> pixel red channel == 10.
        assert image.getpixel((0, 0))[0] == 10, (
            f'expected middle-slice pixel value 10, got {image.getpixel((0, 0))}'
        )

        # Fail-fast error path: corrupt the CSV to reference a missing volume.
        bad_row = pd.DataFrame([{
            'folder': 'AMD', 'oct_imgname': 'AMD_999_%02d', 'slice_indices': '0-1-2',
            'label': 0, 'split': 'train',
        }])
        bad_csv = tmp_root / 'fold0_bad.csv'
        bad_row.to_csv(bad_csv, index=False)
        bad_ds = PublicOCTBscanDataset(bad_csv, tmp_root, split='train')
        try:
            bad_ds[0]
            raise AssertionError('expected FileNotFoundError for missing image')
        except FileNotFoundError as e:
            assert 'AMD_999_01' in str(e), f'error message missing attempted filename: {e}'

        print('ALL CHECKS PASSED')
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == '__main__':
    main()
```

Run: `python verify_dataset_public_oct.py` (from the scratchpad directory)
Expected: `ModuleNotFoundError: No module named 'mutils.dataset_public_oct'`

- [ ] **Step 2: Create `mutils/dataset_public_oct.py`**

```python
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


"""CSV/fold-driven dataset for OphFoundation's public OCT B-scan benchmark
datasets (duke14, glaucoma, oimhs, umn), compatible with the fold-split CSV
schema confirmed against a live sample (duke14_fold_split0.csv).

Expected CSV columns (same schema across all 4 datasets):
    - folder: per-class subdirectory under `root_dir` (e.g. 'AMD').
    - oct_imgname: per-volume filename template containing a '%02d'-style
      placeholder for the slice index, e.g. 'AMD_1_%02d'.
    - slice_indices: dash-separated slice indices for the volume,
      e.g. '0-1-2-...-48'.
    - label: integer class label.
    - split: one of 'train', 'val', 'test'.

Each OCT slice file is expected at:
    `root_dir/folder/{oct_imgname % slice_index}{extension}`
where `extension` is auto-detected among IMAGE_EXTENSIONS. The on-disk
naming convention was not directly verified against a live directory
listing, so this list is a best-effort guess -- if it's wrong,
_resolve_image_path raises a FileNotFoundError with enough detail
(attempted paths, actual directory contents) to fix it quickly rather than
failing silently or deep inside a DataLoader worker.
"""

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def _middle_slice_index(row) -> int:
    slice_indices = str(row['slice_indices']).split('-')
    return int(slice_indices[len(slice_indices) // 2])


def _resolve_image_path(root_dir: Path, row) -> Path:
    slice_index = _middle_slice_index(row)
    folder_dir = root_dir / str(row['folder'])
    base_name = str(row['oct_imgname']) % slice_index
    for ext in IMAGE_EXTENSIONS:
        candidate = folder_dir / f'{base_name}{ext}'
        if candidate.exists():
            return candidate
    tried = [str(folder_dir / f'{base_name}{ext}') for ext in IMAGE_EXTENSIONS]
    try:
        listing = sorted(p.name for p in folder_dir.iterdir())[:20]
    except FileNotFoundError:
        listing = [f'<folder does not exist: {folder_dir}>']
    raise FileNotFoundError(
        f"Could not find image for row with base name '{base_name}'.\n"
        "Tried:\n  " + "\n  ".join(tried) + "\n"
        f"First 20 entries actually in {folder_dir}:\n  " + "\n  ".join(listing)
    )


class PublicOCTBscanDataset(Dataset):
    """Middle-slice-per-volume dataset for OphFoundation's public OCT
    B-scan fold CSVs (duke14, glaucoma, oimhs, umn).

    MIRAGE's classification head only accepts a single 2D image per sample,
    so -- like `mutils/dataset_uf.py`'s `UFCohortDataset` -- each volume
    (one CSV row) collapses to its middle slice.
    """

    def __init__(self, csv_file, root_dir, split, transform=None):
        data_frame = pd.read_csv(csv_file)
        self.data_frame = data_frame[data_frame['split'] == split].reset_index(drop=True)
        print(f'>>> PublicOCTBscanDataset [{split}]: {len(self.data_frame)} samples')
        print('Label distribution:\n', self.data_frame['label'].value_counts())
        self.targets = self.data_frame['label'].tolist()
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        img_path = _resolve_image_path(self.root_dir, row)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row['label'])
```

- [ ] **Step 3: Run the verification script and confirm it passes**

Run: `python verify_dataset_public_oct.py` (from the scratchpad directory)
Expected: `ALL CHECKS PASSED`

- [ ] **Step 4: Commit**

```bash
git add mutils/dataset_public_oct.py
git commit -m "$(cat <<'EOF'
Add PublicOCTBscanDataset for CSV/fold-driven OCT bscan loading

Middle-slice-per-volume dataset reading OphFoundation's public OCT
fold-split CSV schema, confirmed against a live duke14 sample row.
Sibling to mutils/dataset_uf.py's UFCohortDataset.
EOF
)"
```

---

### Task 2: Wire the new loader into `run_cls_tuning_bscan.py`

**Files:**
- Modify: `run_cls_tuning_bscan.py:1-18` (module docstring)
- Modify: `run_cls_tuning_bscan.py:19-42` (imports)
- Modify: `run_cls_tuning_bscan.py:116-129` (add `--fold`)
- Modify: `run_cls_tuning_bscan.py:211-230` (`--csv_root`, `--data_set` choices)
- Modify: `run_cls_tuning_bscan.py:235-265` (`process_args`)
- Modify: `run_cls_tuning_bscan.py:268-283` (`get_output_dir`)
- Modify: `run_cls_tuning_bscan.py:286-290` (`build_dataset`)
- Verification script (not committed): `<scratchpad>/verify_bscan_process_args.py`

- [ ] **Step 1: Replace the module docstring**

In `run_cls_tuning_bscan.py`, replace lines 1-18:

```python
"""Public OCT B-scan dataset classification tuning.

Dedicated entry point for OphFoundation's public OCT B-scan benchmark
datasets (duke14, glaucoma, oimhs, umn), referenced from OphFoundation's own
finetune-UF-benchmark_*_single.sh scripts.

Unlike run_cls_tuning_fundus.py (plain train/val/test/Class_x/ image
folders), these 4 datasets' raw data is volumetric: one CSV row per volume,
with a '%02d'-style filename template and a dash-separated list of slice
indices (see mutils/dataset_public_oct.py for the exact schema). This script
reads that CSV/fold convention directly via PublicOCTBscanDataset, which
selects the middle slice of each volume as its one representative 2D image
-- consistent with MIRAGE's classification head accepting a single image
per sample. --fold selects which pre-computed fold-split CSV to use
(default 0); looping over multiple folds is left to the caller (see
run_bscan_all_tasks.sh), not done automatically by this script.
"""
```

- [ ] **Step 2: Update imports and add the per-dataset path config**

Replace this block (current lines 19-42):

```python
from typing import Callable
from copy import deepcopy
import json
import os
import sys
import hashlib

import argparse
import datetime
import pandas as pd
import time
from pathlib import Path
import socket

import torch
from torch.utils.data import DataLoader

from timm.loss import LabelSmoothingCrossEntropy
from torchvision import datasets

from mutils import misc
from mutils.classification import train_1_epoch, evaluate, EarlyStopping
from mutils.misc import fix_seeds, SortingHelpFormatter
from fm_cls_config import fm_config_factory
```

with:

```python
from typing import Callable
from copy import deepcopy
import json
import os
import sys
import hashlib

import argparse
import datetime
import pandas as pd
import time
from pathlib import Path
import socket

import torch
from torch.utils.data import DataLoader

from timm.loss import LabelSmoothingCrossEntropy

from mutils import misc
from mutils.classification import train_1_epoch, evaluate, EarlyStopping
from mutils.dataset_public_oct import PublicOCTBscanDataset
from mutils.misc import fix_seeds, SortingHelpFormatter
from fm_cls_config import fm_config_factory


# Per-dataset raw image subdirectory (under --data_root) and fold-split CSV
# location (under --csv_root), per OphFoundation's
# finetune-UF-benchmark_*_single.sh reference scripts.
PUBLIC_OCT_DATASETS = {
    'duke14': {
        'image_subdir': 'DUKE_14_Srin/duke14_processed',
        'csv_subdir': 'finetune_duke14_fewshot_3D_10folds_effective_fold',
        'csv_pattern': 'duke14_fold_split{fold}.csv',
    },
    'glaucoma': {
        'image_subdir': 'GLAUCOMA/glaucoma_processed',
        'csv_subdir': 'finetune_glaucoma_fewshot_3D_10folds_correct_visit',
        'csv_pattern': 'glaucoma_fold_{fold}_split.csv',
    },
    'oimhs': {
        'image_subdir': 'OIMHS_dataset/cls_images',
        'csv_subdir': 'finetune_oimhs_fewshot_3D_10folds_correct_',
        'csv_pattern': 'oimhs_fold_{fold}_split_ref.csv',
    },
    'umn': {
        'image_subdir': 'UMN/UMN_dataset/image_classification',
        'csv_subdir': 'finetune_umn_fewshot_3D_10folds_correct',
        'csv_pattern': 'umn_fold_{fold}_split_ref.csv',
    },
}
```

Note: `datasets` (torchvision) import removed since `ImageFolder` is no longer used; `PublicOCTBscanDataset` and `PUBLIC_OCT_DATASETS` added.

- [ ] **Step 3: Add `--fold`**

In `run_cls_tuning_bscan.py`, find this block (current lines 135-138):

```python
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed for reproducibility. (default: %(default)s)',
    )
```

Add immediately after it:

```python
    parser.add_argument(
        '--fold', default=0, type=int,
        help='Which pre-computed fold-split CSV to use (0-9).'
            ' (default: %(default)s)',
    )
```

- [ ] **Step 4: Add `--csv_root` and constrain `--data_set` to known datasets**

Replace this block (current lines 218-230):

```python
    # Dataset parameters
    required_parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory for the classification datasets. (required)',
    )
    required_parser.add_argument(
        '--data_set',
        type=str,
        required=True,
        help='Dataset directory name. (required)',
    )
```

with:

```python
    # Dataset parameters
    required_parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing the raw per-dataset image folders'
            ' (e.g. .../OCTCubeM/assets/ext_oph_datasets/). (required)',
    )
    required_parser.add_argument(
        '--csv_root',
        type=str,
        required=True,
        help='Root directory containing the per-dataset fold-split CSV'
            ' directories (e.g. .../OphFoundation/Public_OCT_split/).'
            ' (required)',
    )
    required_parser.add_argument(
        '--data_set',
        type=str,
        required=True,
        choices=sorted(PUBLIC_OCT_DATASETS.keys()),
        help='Dataset name. (required)',
    )
```

- [ ] **Step 5: Rewrite `process_args`**

Replace the whole function (current lines 235-265):

```python
def process_args(args):
    hostname = socket.gethostname()
    print(f'Running on {hostname}')

    if args.data_root[-1] != '/':
        args.data_root += '/'
    args.data_path = args.data_root + args.data_set

    # Automatic number of classes calculation
    train_data_path = args.data_path + '/train'
    num_classes = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_classes += 1
    num_samples = 0
    for class_dir in Path(train_data_path).iterdir():
        if class_dir.is_dir():
            num_samples += len(list(class_dir.iterdir()))
    args.num_classes = num_classes
    print(f'Number of classes: {num_classes}')
    print(f'Number of training samples: {num_samples}')

    if args.batch_size is None:
        # Automatic batch size calculation
        # Batch size is closest power of 2 to 1/10 of the dataset, with a
        # maximum of 64.
        args.batch_size = min(64, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
        if args.batch_size < 1:
            args.batch_size = 8
    print(f'Batch size: {args.batch_size}')
    return args
```

with:

```python
def process_args(args):
    hostname = socket.gethostname()
    print(f'Running on {hostname}')

    if args.data_root[-1] != '/':
        args.data_root += '/'
    if args.csv_root[-1] != '/':
        args.csv_root += '/'

    dataset_config = PUBLIC_OCT_DATASETS[args.data_set]
    args.image_root = args.data_root + dataset_config['image_subdir']
    csv_dir = args.csv_root + dataset_config['csv_subdir']
    args.csv_file = csv_dir + '/' + dataset_config['csv_pattern'].format(fold=args.fold)

    # Automatic number of classes calculation from the CSV's label column
    data_frame = pd.read_csv(args.csv_file)
    train_frame = data_frame[data_frame['split'] == 'train']
    args.num_classes = int(train_frame['label'].nunique())
    num_samples = len(train_frame)
    print(f'Number of classes: {args.num_classes}')
    print(f'Number of training samples: {num_samples}')

    if args.batch_size is None:
        # Automatic batch size calculation
        # Batch size is closest power of 2 to 1/10 of the dataset, with a
        # maximum of 64.
        args.batch_size = min(64, 2 ** (int(round(num_samples * 0.25)).bit_length() - 1))
        if args.batch_size < 1:
            args.batch_size = 8
    print(f'Batch size: {args.batch_size}')
    return args
```

- [ ] **Step 6: Update `get_output_dir` to separate fold results**

Replace the function (current lines 268-283):

```python
def get_output_dir(args, model_name):
    # Set output directory based on some arguments
    output_dir = args.base_output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.version}/'
    output_dir += f'{args.seed}/'
    output_dir += f'{args.data_set}/'
    output_dir += f'{model_name}'
    if args.linear_probing:
        output_dir += '_linear'
    else:
        output_dir += '_finetune'
    if args.weights is not None:
        output_dir += '_w'
    return output_dir
```

with:

```python
def get_output_dir(args, model_name):
    # Set output directory based on some arguments
    output_dir = args.base_output_dir
    if output_dir[-1] != '/':
        output_dir += '/'
    output_dir += f'{args.version}/'
    output_dir += f'{args.seed}/'
    output_dir += f'{args.data_set}/'
    if args.fold != 0:
        output_dir += f'fold{args.fold}/'
    output_dir += f'{model_name}'
    if args.linear_probing:
        output_dir += '_linear'
    else:
        output_dir += '_finetune'
    if args.weights is not None:
        output_dir += '_w'
    return output_dir
```

- [ ] **Step 7: Rewrite `build_dataset`**

Replace the function (current lines 286-290):

```python
def build_dataset(subset, args, build_transform: Callable, augment=False):
    transform = build_transform(subset, augment)
    root = os.path.join(args.data_path, subset)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset
```

with:

```python
def build_dataset(subset, args, build_transform: Callable, augment=False):
    transform = build_transform(subset, augment)
    dataset = PublicOCTBscanDataset(
        csv_file=args.csv_file,
        root_dir=args.image_root,
        split=subset,
        transform=transform,
    )
    return dataset
```

- [ ] **Step 8: Write an integration verification script and confirm it fails first**

Before Step 2-7 edits are made this would fail with `AttributeError` (no `csv_root`), but since steps are applied in order, write and run this *after* Step 7. Create `<scratchpad>/verify_bscan_process_args.py`:

```python
"""Integration smoke test for run_cls_tuning_bscan.py's process_args/
build_dataset against synthetic data (no real cluster access needed).
Run with `python verify_bscan_process_args.py` from the repo root.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, r'f:\github\MIRAGE')

from run_cls_tuning_bscan import process_args, build_dataset, get_output_dir


def make_synthetic_dataset(data_root: Path, csv_root: Path):
    image_subdir = 'DUKE_14_Srin/duke14_processed'
    csv_subdir = 'finetune_duke14_fewshot_3D_10folds_effective_fold'
    class_dir = data_root / image_subdir / 'AMD'
    class_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for patient_num, split in [(1, 'train'), (2, 'val'), (3, 'test')]:
        base = f'AMD_{patient_num}_%02d'
        for slice_idx in (0, 1, 2):
            Image.new('RGB', (8, 8)).save(class_dir / (base % slice_idx + '.jpg'))
        rows.append({
            'folder': 'AMD', 'oct_imgname': base, 'slice_indices': '0-1-2',
            'label': 0, 'split': split,
        })
    csv_dir = csv_root / csv_subdir
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_dir / 'duke14_fold_split0.csv', index=False)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='bscan_integration_'))
    try:
        data_root = tmp / 'data'
        csv_root = tmp / 'csv'
        make_synthetic_dataset(data_root, csv_root)

        args = argparse.Namespace(
            data_root=str(data_root), csv_root=str(csv_root), data_set='duke14',
            fold=0, batch_size=None,
        )
        args = process_args(args)
        assert args.num_classes == 1, f'expected 1 class, got {args.num_classes}'
        assert args.batch_size == 8, f'expected fallback batch_size 8, got {args.batch_size}'

        train_ds = build_dataset('train', args, build_transform=lambda subset, augment: None)
        assert len(train_ds) == 1, f'expected 1 train sample, got {len(train_ds)}'

        out_args = argparse.Namespace(
            base_output_dir='/tmp/out', version='v1', seed=0, data_set='duke14',
            fold=0, linear_probing=True, weights='MIRAGE-Base.pth',
        )
        assert get_output_dir(out_args, 'MIRAGE-Base') == '/tmp/out/v1/0/duke14/MIRAGE-Base_linear_w'
        out_args.fold = 3
        assert get_output_dir(out_args, 'MIRAGE-Base') == '/tmp/out/v1/0/duke14/fold3/MIRAGE-Base_linear_w'

        print('ALL CHECKS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
```

Run: `python verify_bscan_process_args.py` (from the scratchpad directory)
Expected: `ALL CHECKS PASSED` (all edits from Steps 1-7 must be in place first)

- [ ] **Step 9: Sanity-check the CLI still parses**

Run: `python run_cls_tuning_bscan.py --help`
Expected: help text prints without a Python traceback, showing `--csv_root`, `--fold`, and `--data_set` with `choices: {duke14,glaucoma,oimhs,umn}`.

- [ ] **Step 10: Commit**

```bash
git add run_cls_tuning_bscan.py
git commit -m "$(cat <<'EOF'
Switch run_cls_tuning_bscan.py to CSV/fold-driven data loading

Replaces the ImageFolder-over-nonexistent-train/val/test assumption
with PublicOCTBscanDataset, reading each dataset's real on-disk layout
(raw per-slice volume folders + a fold-split CSV) directly. Adds
--csv_root and --fold; --data_set is now constrained to the 4 known
datasets via a PUBLIC_OCT_DATASETS path config.
EOF
)"
```

---

### Task 3: Update the bscan shell scripts

**Files:**
- Modify: `run_bscan_all_tasks.sh:15-54`
- Modify: `run_bscan_multirun.sh:23-62`

- [ ] **Step 1: Update `run_bscan_all_tasks.sh`**

Replace lines 15-54:

```bash
# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public OCT B-scan datasets (see docs/classification_benchmark.md).
#   IMPORTANT: OphFoundation's own raw data for these datasets is volumetric
#   (one CSV row per volume, dynamically resampled to 20 slices per volume --
#   see util/datasets_oct_pub.py's OCT3D_DUKE_Dataset/_select_fixed_count in
#   the OphFoundation repo). Each Class_x/ file here must be ONE selected
#   slice per volume, not every raw slice PNG -- pointing this at the raw
#   per-slice folders directly would make ImageFolder treat every slice as
#   its own independent sample and break the intended train/val/test split
#   by volume/patient.
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_bscan_public/"
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark these same public OCT datasets via
#   their own CSV/fold pipeline). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from the folder structure, not passed here.
DATASETS=(duke14 glaucoma oimhs umn)

# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: DATASET
launch() {
    local PROBE_FLAG=$1
    local DATASET=$2
    ./runner python run_cls_tuning_bscan.py \
        --runners 2 \
        -- \
        --version v1 \
        --seed 0 \
        --weights \
            $WEIGHTS_BASE \
            $WEIGHTS_LARGE \
        $PROBE_FLAG \
        --data_root \
            $DATA_ROOT \
        --data_set \
            $DATASET \
        --base_output_dir \
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan &
}
```

with:

```bash
# Root containing the raw per-dataset OCT volume/slice image folders (one
#   subfolder per class, e.g. duke14_processed/{AMD,DME,NORMAL}) --
#   see mutils/dataset_public_oct.py for the exact per-dataset subdirectory
#   layout and CSV/fold-split convention. --fold 0 uses the first
#   pre-computed train/val/test partition; looping over multiple folds is
#   not done here (staged rollout -- add a fold loop only once fold 0 is
#   validated).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"
CSV_ROOT="/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"
FOLD=0
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from each dataset's fold-split CSV, not passed
#   here.
DATASETS=(duke14 glaucoma oimhs umn)

# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: DATASET
launch() {
    local PROBE_FLAG=$1
    local DATASET=$2
    ./runner python run_cls_tuning_bscan.py \
        --runners 2 \
        -- \
        --version v1 \
        --seed 0 \
        --weights \
            $WEIGHTS_BASE \
            $WEIGHTS_LARGE \
        $PROBE_FLAG \
        --data_root \
            $DATA_ROOT \
        --csv_root \
            $CSV_ROOT \
        --fold \
            $FOLD \
        --data_set \
            $DATASET \
        --base_output_dir \
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan &
}
```

- [ ] **Step 2: Update `run_bscan_multirun.sh`**

Replace lines 23-62:

```bash
# Datasets follow MIRAGE's "public dataset setting": pre-split
#   train/val/test/Class_x/ image folders under --data_root (see
#   docs/classification_benchmark.md). IMPORTANT: OphFoundation's own raw
#   data is volumetric (one CSV row per volume, resampled to 20 slices at
#   load time -- see the caveat in run_cls_tuning_bscan.py's module
#   docstring). Each Class_x/ file here must be ONE selected slice per
#   volume, not every raw slice PNG.
LINEAR_PROBING=${1:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public OCT B-scan datasets (see docs/classification_benchmark.md).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_bscan_public/"

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark these same public OCT datasets via
#   their own CSV/fold pipeline). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from the folder structure, not passed here.
./runner python run_cls_tuning_bscan.py \
    --runners 8 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth \
    $LINEAR_PROBING_FLAG \
    --data_root \
        $DATA_ROOT \
    --data_set \
        duke14 \
        glaucoma \
        oimhs \
        umn \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan
exit
```

with:

```bash
# Datasets follow the CSV/fold-split convention (see
#   mutils/dataset_public_oct.py): raw per-slice volume folders under
#   --data_root, plus a fold-split CSV under --csv_root that assigns each
#   volume to train/val/test and gives its integer label. --fold 0 uses the
#   first pre-computed partition (see run_cls_tuning_bscan.py's module
#   docstring for why looping over folds isn't done automatically).
LINEAR_PROBING=${1:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# Root containing the raw per-dataset OCT volume/slice image folders.
DATA_ROOT="/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"
CSV_ROOT="/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"
FOLD=0

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from each dataset's fold-split CSV, not passed
#   here.
./runner python run_cls_tuning_bscan.py \
    --runners 8 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth \
    $LINEAR_PROBING_FLAG \
    --data_root \
        $DATA_ROOT \
    --csv_root \
        $CSV_ROOT \
    --fold \
        $FOLD \
    --data_set \
        duke14 \
        glaucoma \
        oimhs \
        umn \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan
exit
```

- [ ] **Step 3: Syntax-check both scripts**

Run: `bash -n run_bscan_all_tasks.sh && bash -n run_bscan_multirun.sh && echo OK`
Expected: `OK` (no output before it)

- [ ] **Step 4: Commit**

```bash
git add run_bscan_all_tasks.sh run_bscan_multirun.sh
git commit -m "$(cat <<'EOF'
Point bscan shell scripts at the real CSV/fold data layout

DATA_ROOT now targets the OCTCubeM assets base (raw per-slice volume
folders) instead of the nonexistent MIRAGE_data/cls_bscan_public/;
adds --csv_root (Public_OCT_split base) and --fold 0.
EOF
)"
```

---

### Task 4: Update documentation

**Files:**
- Modify: `docs/classification_tuning.md:60-84`

- [ ] **Step 1: Rewrite the "Public fundus and OCT B-scan benchmark scripts" section**

Replace lines 60-84:

```markdown
## Public fundus and OCT B-scan benchmark scripts

`run_cls_tuning_fundus.py` and `run_cls_tuning_bscan.py` are dedicated
entry points for two families of public benchmark datasets referenced from
[OphFoundation](https://github.com/franciszchen/OphFoundation)'s own
finetuning scripts. Both still use MIRAGE's "public dataset setting"
(pre-split `train/val/test/Class_x/` image folders, see
[docs/classification_benchmark.md](../docs/classification_benchmark.md)) —
not OphFoundation's CSV/fold-split pipeline.

- `run_cls_tuning_fundus.py`: for color fundus photo datasets (e.g.
  `Glaucoma_fundus`, `IDRiD_data`, `JSIEC`, `MESSIDOR2`, `PAPILA`, `Retina`,
  `APTOS2019`). MIRAGE has no dedicated `fundus` domain, so images are
  routed through `slo`, the only other en-face 2D domain — see
  `mirage_wrapper.DOMAIN_CONF`. Its input adapter is single-channel by
  architecture, so this is a cross-domain transfer evaluation, not a
  full-fidelity color setup.
- `run_cls_tuning_bscan.py`: for public OCT B-scan benchmark datasets (e.g.
  `duke14`, `glaucoma`, `oimhs`, `umn`). Uses MIRAGE's default `bscan`
  domain — functionally the same data pipeline as `run_cls_tuning.py`,
  just a dedicated entry point for these datasets.

Both scripts take the same `--weights`/`--data_root`/`--data_set` arguments
as `run_cls_tuning.py`. See `run_fundus.sh`/`run_bscan.sh` for ready-to-edit
copies of these commands.
```

with:

```markdown
## Public fundus and OCT B-scan benchmark scripts

`run_cls_tuning_fundus.py` and `run_cls_tuning_bscan.py` are dedicated
entry points for two families of public benchmark datasets referenced from
[OphFoundation](https://github.com/franciszchen/OphFoundation)'s own
finetuning scripts. They use two *different* data conventions, matching how
each family is actually organized on disk:

- `run_cls_tuning_fundus.py`: for color fundus photo datasets (e.g.
  `Glaucoma_fundus`, `IDRiD_data`, `JSIEC`, `MESSIDOR2`, `PAPILA`, `Retina`,
  `APTOS2019`). Uses MIRAGE's "public dataset setting" — pre-split
  `train/val/test/Class_x/` image folders under `--data_root` (see
  [docs/classification_benchmark.md](../docs/classification_benchmark.md)),
  loaded with plain `ImageFolder`. MIRAGE has no dedicated `fundus` domain,
  so images are routed through `slo`, the only other en-face 2D domain —
  see `mirage_wrapper.DOMAIN_CONF`. Its input adapter is single-channel by
  architecture, so this is a cross-domain transfer evaluation, not a
  full-fidelity color setup.
- `run_cls_tuning_bscan.py`: for public OCT B-scan benchmark datasets
  `duke14`, `glaucoma`, `oimhs`, `umn`. These are **not** pre-split image
  folders — the raw data is volumetric (one CSV row per volume) — so this
  script reads OphFoundation's own CSV/fold-split convention directly via
  `mutils/dataset_public_oct.py`'s `PublicOCTBscanDataset`, which selects
  the middle slice of each volume as its one representative 2D image
  (MIRAGE's classification head only accepts a single image per sample).
  Takes `--data_root` (raw per-dataset image folders, e.g.
  `.../OCTCubeM/assets/ext_oph_datasets/`), `--csv_root` (fold-split CSV
  base, e.g. `.../OphFoundation/Public_OCT_split/`), and `--fold` (default
  `0`; selects which pre-computed train/val/test partition to use — looping
  over all 10 folds is left to the caller, see `run_bscan_all_tasks.sh`).
  Uses MIRAGE's default `bscan` domain.

See `run_fundus.sh`/`run_bscan.sh` for ready-to-edit copies of these
commands.
```

- [ ] **Step 2: Commit**

```bash
git add docs/classification_tuning.md
git commit -m "$(cat <<'EOF'
Document run_cls_tuning_bscan.py's CSV/fold data convention

Corrects the doc, which still described the old (nonexistent)
pre-split-folder assumption; distinguishes it from
run_cls_tuning_fundus.py, which still uses plain ImageFolder.
EOF
)"
```

---

## Self-review notes

- **Spec coverage**: Task 1 covers spec decision 1 (dataset class) and the
  unconfirmed-extension risk (fail-fast error, verified in Step 1's test).
  Task 2 covers decisions 2 (path config) and 3 (`run_cls_tuning_bscan.py`
  changes, including fold-aware output dir). Task 3 covers decisions 4-5
  (fold scope, shell scripts). Task 4 covers decision 6 (docs).
- **`run_fundus.sh`/`run_bscan.sh`** (singular, non-"all-tasks"/"multirun"
  variants mentioned in `docs/classification_tuning.md`) are out of scope
  for this plan — they weren't part of the original bug report and aren't
  in the `git status`/recent-commits context; if they exist and also
  hardcode the old `DATA_ROOT`, that's a follow-up, not blocking this plan.
- **Type/signature consistency**: `PublicOCTBscanDataset(csv_file, root_dir,
  split, transform=None)` signature matches its two call sites (Task 1's
  verification script and Task 2 Step 7's `build_dataset`). `PUBLIC_OCT_DATASETS`
  keys (`duke14`, `glaucoma`, `oimhs`, `umn`) match `--data_set`'s `choices`
  and both shell scripts' `DATASETS`/`--data_set` values.
