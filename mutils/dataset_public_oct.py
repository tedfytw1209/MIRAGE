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
