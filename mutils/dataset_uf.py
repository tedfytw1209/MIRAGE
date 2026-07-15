from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


"""CSV-driven datasets for the UF (University of Florida) ophthalmology
cohort, compatible with OphFoundation's dataset schema
(see `util/dataset_uf.py` in the OphFoundation repository).

Expected CSV columns:
    - folder: per-visit subdirectory under `root_dir`.
    - fundus_imgname: SLO en-face image filename inside `folder`
      ('fundus' is a naming holdover from OphMAE/OphFoundation; the image
      itself is an SLO scan, not a color fundus photo).
    - oct_imgname: base name of the OCT B-scan slice files.
    - eye: laterality ('OD'/'OS', with or without the 'lat' prefix).
    - slice_indices: dash-separated slice indices, e.g. "12-13-14".
    - label: integer class label.
    - split: one of 'train', 'val', 'test'.

Each OCT slice file is expected at:
    `root_dir/folder/{oct_imgname}_{lat+eye}_{slice_index}.jpg`
"""


def _bscan_image_path(root_dir: Path, row) -> Path:
    eye = str(row['eye'])
    if 'lat' not in eye:
        eye = 'lat' + eye
    slice_indices = str(row['slice_indices']).split('-')
    # Middle slice of the volume (closest to the fovea in most captures).
    slice_index = slice_indices[len(slice_indices) // 2]
    img_name = f"{row['oct_imgname']}_{eye}_{slice_index}.jpg"
    return root_dir / row['folder'] / img_name


def _slo_image_path(root_dir: Path, row) -> Path:
    return root_dir / row['folder'] / row['fundus_imgname']


class UFCohortDataset(Dataset):
    """Single-modality UF cohort dataset.

    MIRAGE's classification head only accepts a single 2D image per sample
    (see `mirage_wrapper.MIRAGEClsGlobal`, pre-multimodal-support version).
    `modality='bscan'` (default) picks the middle slice of `slice_indices`
    as that image; `modality='slo'` uses the SLO scan (the CSV's
    `fundus_imgname` column) instead.
    """

    def __init__(self, csv_file, root_dir, split, transform=None, modality='bscan'):
        assert modality in ('bscan', 'slo'), f'Unknown modality: {modality}'
        data_frame = pd.read_csv(csv_file)
        self.data_frame = data_frame[data_frame['split'] == split].reset_index(drop=True)
        print(f'>>> UFCohortDataset [{split}, {modality}]: {len(self.data_frame)} samples')
        print('Label distribution:\n', self.data_frame['label'].value_counts())
        self.targets = self.data_frame['label'].tolist()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.modality = modality

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        if self.modality == 'bscan':
            img_path = _bscan_image_path(self.root_dir, row)
        else:
            img_path = _slo_image_path(self.root_dir, row)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row['label'])


class UFCohortMultiModalDataset(Dataset):
    """True multi-modal (OCT B-scan + SLO) UF cohort dataset.

    Returns `({'bscan': image, 'slo': image}, label)` per sample, for use
    with a MIRAGE classifier configured with `in_domains=['bscan', 'slo']`
    (see `mirage_wrapper.MIRAGEClsGlobal.forward`, which accepts a dict of
    per-domain tensors and jointly attends to both through the shared
    encoder before pooling). `transform_bscan`/`transform_slo` are applied
    independently, so any random augmentation they include is sampled
    separately per modality.
    """

    def __init__(self, csv_file, root_dir, split, transform_bscan=None, transform_slo=None):
        data_frame = pd.read_csv(csv_file)
        self.data_frame = data_frame[data_frame['split'] == split].reset_index(drop=True)
        print(f'>>> UFCohortMultiModalDataset [{split}]: {len(self.data_frame)} samples')
        print('Label distribution:\n', self.data_frame['label'].value_counts())
        self.targets = self.data_frame['label'].tolist()
        self.root_dir = Path(root_dir)
        self.transform_bscan = transform_bscan
        self.transform_slo = transform_slo

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]

        bscan = Image.open(_bscan_image_path(self.root_dir, row)).convert('RGB')
        if self.transform_bscan is not None:
            bscan = self.transform_bscan(bscan)

        slo = Image.open(_slo_image_path(self.root_dir, row)).convert('RGB')
        if self.transform_slo is not None:
            slo = self.transform_slo(slo)

        return {'bscan': bscan, 'slo': slo}, int(row['label'])
