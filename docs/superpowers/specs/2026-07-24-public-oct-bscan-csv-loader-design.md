# Design: CSV/fold-driven data loading for the 4 public OCT B-scan datasets

Date: 2026-07-24
Status: Approved

## Context

`run_cls_tuning_bscan.py` was built per the 2026-07-20 spec
(`docs/superpowers/specs/2026-07-20-public-fundus-bscan-cls-scripts-design.md`)
as a plain `ImageFolder` loader over `data_root/data_set/{train,val,test}/Class_x/`,
identical in shape to `run_cls_tuning_fundus.py`. That spec explicitly declared
per-volume/slice-grouping logic a non-goal: "picking a representative slice is
the user's data-prep step, done before these scripts run."

In practice, no such pre-split data exists for the 4 target datasets (duke14,
glaucoma, oimhs, umn) — `run_bscan_all_tasks.sh` pointed `--data_root` at
`/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_bscan_public/`, which is empty/
nonexistent, causing every job to fail with `FileNotFoundError` on `.../train`.

Investigation of OphFoundation's reference scripts
(`finetune-UF-benchmark_{duke14,glaucoma,oimhs,umn}_single.sh`) showed the real
data for these 4 datasets is organized completely differently: raw per-slice
image folders (one subfolder per class, e.g. `duke14_processed/{AMD,DME,NORMAL}`)
plus a CSV file per fold that lists one row per volume, with columns:

```
folder,imgname,eye,slice_indices,slice_num,fundus_imgname,patient_id,
oct_imgname,OCT,depth,oct_valid,oct_img_size,fundus_valid,fundus_img_size,
label,Glaucoma_icd,icd_eye,split
```

Confirmed (user-provided sample row from
`duke14_fold_split0.csv`) and confirmed to be the same schema across all 4
datasets. Key fields:
- `folder`: class subdirectory name (e.g. `AMD`).
- `oct_imgname`: a literal `%02d`-style Python format template per volume
  (e.g. `"AMD_1_%02d"`), not a fixed filename.
- `slice_indices`: dash-separated list of all slice indices in the volume
  (e.g. `"0-1-2-...-48"`).
- `label`: integer class label, given directly (not inferred from `folder`).
- `split`: `train`/`val`/`test`, given directly per volume (no numeric
  fold-column arithmetic needed — each `*_fold_N.csv` file is simply one
  pre-computed train/val/test partition).

This design reverses the 2026-07-20 spec's non-goal for `run_cls_tuning_bscan.py`
specifically: instead of requiring a separate, unscoped data-prep step, the
script itself gains CSV/volume-aware loading. `run_cls_tuning_fundus.py` is
unaffected and keeps its plain `ImageFolder` approach — the two scripts now
use genuinely different data conventions, matching how their source datasets
are actually organized on disk.

## Unconfirmed assumption (explicit risk)

The exact on-disk filename convention (extension, and whether `oct_imgname %
slice_index` resolves directly to a file under `root_dir/folder/` with no
further nesting) was not verified against a live directory listing — the user
opted to skip that verification step. The loader must fail fast with a clear,
actionable error (attempted path + directory listing) at dataset-construction
time if this assumption is wrong, rather than failing silently or deep inside
a DataLoader worker.

## Decisions

1. **New dataset class**: `mutils/dataset_public_oct.py`, a
   `PublicOCTBscanDataset(Dataset)`, structurally a sibling to
   `mutils/dataset_uf.py`'s `UFCohortDataset` (same `split`-column filtering
   idiom), but with:
   - Middle-slice selection from `slice_indices` (same "one slice per volume"
     policy already documented as intentional in `run_cls_tuning_bscan.py`'s
     original docstring).
   - Path resolution via the `oct_imgname` `%`-template rather than the UF
     schema's fixed-basename-plus-eye-suffix convention.
   - `num_classes` derived from the CSV's `label` column (`nunique()`), not
     from `iterdir()`.
   - Extension probing (`.jpg`, `.jpeg`, `.png`, in that order) with a
     fail-fast, diagnostic error if none match.

2. **Per-dataset path config**: a small dict in `run_cls_tuning_bscan.py` (same
   pattern as `fm_cls_config.py`'s `fm_config_factory`), since each of the 4
   datasets has an irregular processed-image subdirectory name and CSV
   filename convention:

   ```python
   PUBLIC_OCT_DATASETS = {
       'duke14':   {'image_subdir': 'DUKE_14_Srin/duke14_processed',
                    'csv_subdir': 'finetune_duke14_fewshot_3D_10folds_effective_fold',
                    'csv_pattern': 'duke14_fold_split{fold}.csv'},
       'glaucoma': {'image_subdir': 'GLAUCOMA/glaucoma_processed',
                    'csv_subdir': 'finetune_glaucoma_fewshot_3D_10folds_correct_visit',
                    'csv_pattern': 'glaucoma_fold_{fold}_split.csv'},
       'oimhs':    {'image_subdir': 'OIMHS_dataset/cls_images',
                    'csv_subdir': 'finetune_oimhs_fewshot_3D_10folds_correct_',
                    'csv_pattern': 'oimhs_fold_{fold}_split_ref.csv'},
       'umn':      {'image_subdir': 'UMN/UMN_dataset/image_classification',
                    'csv_subdir': 'finetune_umn_fewshot_3D_10folds_correct',
                    'csv_pattern': 'umn_fold_{fold}_split_ref.csv'},
   }
   ```

3. **`run_cls_tuning_bscan.py` changes**:
   - New args: `--csv_root` (base directory for `Public_OCT_split`, separate
     from `--data_root` which now means the OCTCubeM assets base) and
     `--fold` (default `0`).
   - `process_args()`: derive `num_classes` and sample counts from the CSV
     instead of `iterdir()` over `train/`.
   - `build_dataset()`: construct `PublicOCTBscanDataset` instead of
     `datasets.ImageFolder`.
   - `get_output_dir()`: include the fold number in the output path when
     `fold != 0`, so different folds don't overwrite each other's results.
   - Update the module docstring (currently describes the `ImageFolder`
     assumption and warns about volumetric raw data as something *outside*
     this script's scope) to describe the CSV/volume approach this script
     now implements directly.

4. **Fold scope**: `--fold` is supported as a parameter (so 10-fold CV is
   *possible*), but per the standing staged-rollout preference (validate
   with one configuration before sweeping — see prior multi-seed precedent
   in the UF-cohort sweep scripts), `run_bscan_all_tasks.sh` and
   `run_bscan_multirun.sh` only run **fold 0**. No loop over all 10 folds is
   added until explicitly requested.

5. **Shell scripts**: update `run_bscan_all_tasks.sh` / `run_bscan_multirun.sh`
   to pass `--csv_root` and `--fold 0`, and point `--data_root` at the
   OCTCubeM assets base (e.g. `/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/`)
   instead of the nonexistent `MIRAGE_data/cls_bscan_public/`.

6. **Docs**: update `docs/classification_benchmark.md` and/or
   `docs/classification_tuning.md` to document that `run_cls_tuning_bscan.py`
   uses the CSV/fold convention while `run_cls_tuning_fundus.py` still uses
   plain `ImageFolder` — so the two scripts' differing data conventions are
   explained in one place rather than looking like an inconsistency.

## Explicitly out of scope

- No changes to `run_cls_tuning_fundus.py` or its shell scripts.
- No automatic 10-fold CV driver/loop (single fold only, `--fold 0` default).
- No dual fundus+OCT paired input, no 3D volume resampling beyond the
  existing middle-slice selection, no mode switching (`2d`/`3d`/`dual`), no
  LDAM/Focal/IB/DRW loss balancing — those are OphFoundation reference
  pipeline features that are explicitly not being ported.
- No live verification of on-disk filenames/extensions (flagged as a risk
  above; loader fails fast and clearly if wrong).

## Files touched

- New: `mutils/dataset_public_oct.py`
- Edit: `run_cls_tuning_bscan.py`
- Edit: `run_bscan_all_tasks.sh`
- Edit: `run_bscan_multirun.sh`
- Edit: `docs/classification_benchmark.md` and/or `docs/classification_tuning.md`
