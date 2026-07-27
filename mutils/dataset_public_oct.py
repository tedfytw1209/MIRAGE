import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


"""CSV/fold-driven dataset for OphFoundation's public OCT B-scan benchmark
datasets (duke14, glaucoma, oimhs, umn).

The path-resolution logic below is ported from OphFoundation's own working
loaders (util/datasets_oct_pub.py: OCT3D_DUKE_Dataset,
OCT3D_TemplatePNGDataset, GLAUCOMA_Dataset) rather than guessed, since the
CSV schema and on-disk slice-file naming differ per dataset:

- duke14: columns folder, imgname (fallback oct_imgname), slice_indices,
  label, split. imgname is a '%'-style template, e.g. 'AMD_1_%02d' -- only
  the literal prefix before '%' is used. The on-disk slice number is offset
  by 0 or 1 relative to the CSV's slice index (varies per volume/export
  run) and zero-padded to 3, 2, or 0 digits, saved as '<prefix><n>.png'.
- umn / oimhs: columns folder, imgname, slice_indices, slice_num, label,
  split. imgname already contains a '%d'-style placeholder Python's '%'
  operator can format directly; same 0/1 slice-index offset applies.
- glaucoma: different schema entirely -- patient_id, image_name, image_fmt,
  slice_num, split, label. image_fmt points at a .npy volume (S,H,W or a
  transposed variant) instead of one file per slice.

MIRAGE's classification head takes a single 2D image per sample (see
run_cls_tuning_bscan.py), so where OphFoundation's originals build a full
multi-slice stack, this file resolves just each volume's middle slice.
"""

IMAGE_EXTENSION = '.png'


def _parse_slice_indices(row) -> list:
    slice_indices = row.get('slice_indices', '')
    if isinstance(slice_indices, str) and slice_indices.strip():
        return slice_indices.split('-')
    slice_num = row.get('slice_num', 0)
    slice_num = int(slice_num) if pd.notna(slice_num) else 0
    return [str(i) for i in range(slice_num)]


def _middle_slice(slice_list: list) -> int:
    return int(slice_list[len(slice_list) // 2])


def _choose_offset(candidates_fn, slice_list, k=8) -> int:
    # OphFoundation's fold CSVs number slices 0- or 1-indexed depending on
    # dataset/export run; probe a handful of this volume's slices against
    # both offsets and keep whichever exists on disk more often.
    k = min(k, len(slice_list))
    scores = {0: 0, 1: 0}
    for offset in (0, 1):
        for s in slice_list[:k]:
            if any(p.exists() for p in candidates_fn(int(s) + offset)):
                scores[offset] += 1
    return 0 if scores[0] >= scores[1] else 1


def _raise_not_found(base_dir: Path, tried: list):
    try:
        listing = sorted(p.name for p in base_dir.iterdir())[:20]
    except FileNotFoundError:
        listing = [f'<folder does not exist: {base_dir}>']
    raise FileNotFoundError(
        "Could not find slice image.\nTried:\n  " + "\n  ".join(tried) +
        f"\nFirst 20 entries actually in {base_dir}:\n  " + "\n  ".join(listing)
    )


def _resolve_duke14_path(root_dir: Path, row) -> Path:
    folder = str(row.get('folder', '') or '')
    base_dir = root_dir / folder if folder else root_dir
    template = str(row.get('imgname', row.get('oct_imgname', '')) or '')
    prefix = template.split('%')[0] if '%' in template else template
    if prefix and not prefix.endswith('_'):
        prefix += '_'

    def candidates(slice_index: int):
        return [
            base_dir / f'{prefix}{slice_index:03d}{IMAGE_EXTENSION}',
            base_dir / f'{prefix}{slice_index:02d}{IMAGE_EXTENSION}',
            base_dir / f'{prefix}{slice_index}{IMAGE_EXTENSION}',
        ]

    slice_list = _parse_slice_indices(row)
    middle = _middle_slice(slice_list)
    offset = _choose_offset(candidates, slice_list)
    for path in candidates(middle + offset):
        if path.exists():
            return path
    _raise_not_found(base_dir, [str(p) for p in candidates(middle + offset)])


def _resolve_template_path(root_dir: Path, row) -> Path:
    """umn / oimhs: imgname already has a '%d'-style placeholder."""
    folder = str(row.get('folder', '') or '')
    base_dir = root_dir / folder if folder else root_dir
    template = str(row.get('imgname', '') or '')

    def candidates(slice_index: int):
        stem = template % slice_index
        if not stem.lower().endswith(IMAGE_EXTENSION):
            stem += IMAGE_EXTENSION
        return [base_dir / stem]

    slice_list = _parse_slice_indices(row)
    middle = _middle_slice(slice_list)
    offset = _choose_offset(candidates, slice_list)
    path = candidates(middle + offset)[0]
    if path.exists():
        return path
    _raise_not_found(base_dir, [str(path)])


def _ensure_slice_first(volume: np.ndarray) -> np.ndarray:
    """Normalize a glaucoma .npy volume to (S, H, W)."""
    if volume.ndim == 4:
        if volume.shape[-1] == 1:
            volume = volume[..., 0]
        elif volume.shape[0] == 1:
            volume = volume[0]
        else:
            raise ValueError(f'Unsupported 4D npy shape: {volume.shape}')
    if volume.ndim != 3:
        raise ValueError(f'Unsupported npy shape: {volume.shape}')
    a, b, c = volume.shape
    if c < min(a, b):
        volume = np.transpose(volume, (2, 0, 1))
    return volume


def _to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    x = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)
    x = np.where(finite, x, 0.0)
    mn, mx = float(x.min()), float(x.max())
    if mn >= 0.0 and mx <= 1.5:
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    denom = (mx - mn) if (mx - mn) > 1e-6 else 1.0
    return np.clip((x - mn) / denom * 255.0, 0, 255).astype(np.uint8)


def _load_glaucoma_middle_slice(root_dir: Path, row) -> Image.Image:
    npy_path = row.get('image_fmt', '')
    if not isinstance(npy_path, str) or not npy_path:
        raise FileNotFoundError(f"Empty image_fmt for row: {row.to_dict()}")
    path = Path(npy_path)
    if not path.is_absolute():
        path = root_dir / npy_path
    if not path.exists():
        raise FileNotFoundError(f"npy not found: {path}")
    volume = _ensure_slice_first(np.load(path))
    middle = volume[volume.shape[0] // 2]
    return Image.fromarray(_to_uint8(middle)).convert('RGB')


_RESOLVERS = {
    'duke14': _resolve_duke14_path,
    'umn': _resolve_template_path,
    'oimhs': _resolve_template_path,
}


class PublicOCTBscanDataset(Dataset):
    """Middle-slice-per-volume dataset for OphFoundation's public OCT
    B-scan fold CSVs (duke14, glaucoma, oimhs, umn).

    `dataset_name` selects the per-dataset path-resolution/image-loading
    strategy (see module docstring) -- the 4 datasets do not share a CSV
    schema or on-disk layout, so this cannot be inferred from `root_dir`
    alone.
    """

    def __init__(self, csv_file, root_dir, split, dataset_name, transform=None):
        self.dataset_name = dataset_name
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
        if self.dataset_name == 'glaucoma':
            image = _load_glaucoma_middle_slice(self.root_dir, row)
        else:
            img_path = _RESOLVERS[self.dataset_name](self.root_dir, row)
            image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row['label'])
