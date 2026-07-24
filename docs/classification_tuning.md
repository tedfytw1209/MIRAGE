# Classification tuning

This repository provides the code to tune MIRAGE and other state-of-the-art foundation models for OCT and SLO classification tasks.


## Requirements

See [README#Requirements](../README.md#requirements) for the requirements.


## Data

The OCT segmentation datasets are available in [docs/classification_benchmark.md](../docs/classification_benchmark.md).


## Usage

The script `run_cls_tuning.py` provides the main entry point to tune the models. It supports several command-line arguments to configure the training process.

We also provide the utility script `runner` (see [docs/runner.md](../docs/runner.md)) to run multiple experiments easily by specifying multiple entries for the same argument.
Below we provide an example to tune MIRAGE (both Base and Large) on the Duke DME dataset.


```bash
./runner python run_cls_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --linear_probing \
    --data_root \
        ./__datasets/Classification \
    --data_set \
        GAMMA
```

> [!TIP]
> Run the script with the `-h` or `--help` flag to see the available options.


By default, the script will save the model weights and the training logs in the `./__output/cls` directory.
You can specify a different output directory using the `--base_output_dir` argument.

> [!IMPORTANT]
> The script uses the filename of the weights to determine which model configuration to use. In particular, the filename should contain the model name, so that the following substrings load the corresponding model configuration (case-insensitive):
>
> - `mirage-base`: MIRAGE-Base
> - `mirage-large`: MIRAGE-Large



## Adding a new dataset

To add a new dataset, you need to respect the dataset structure indicated in [docs/classification_benchmark.md](../docs/classification_benchmark.md).


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


## UF cohort (CSV-driven datasets)

The UF cohort dataset (and any dataset following the same CSV schema, as used
by [OphFoundation](https://github.com/franciszchen/OphFoundation)) is not
laid out as `train/val/test/Class_x/` image folders, so it is tuned with a
separate entry point, `run_cls_tuning_UF.py`, instead of `run_cls_tuning.py`.

Each sample is described by a row in a CSV with columns `folder`,
`fundus_imgname`, `oct_imgname`, `eye`, `slice_indices`, `label`, and `split`
(`train`/`val`/`test`). MIRAGE's classification head only accepts a single 2D
image per sample (see `mirage_wrapper.py`), so `mutils/dataset_uf.py`
collapses each row to one image:

- `--uf_modality bscan` (default): the middle slice of `slice_indices`, fed
  through MIRAGE's `bscan` input.
- `--uf_modality slo`: the image in the CSV's `fundus_imgname` column, fed
  through MIRAGE's `slo` input. Despite the column name, this is an SLO
  scan, not a color fundus photo — `fundus_imgname` is a naming holdover
  from OphMAE/OphFoundation — so this is an in-domain comparison too.

```bash
./runner python run_cls_tuning_UF.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --linear_probing \
    --data_root \
        /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ \
    --csv_file_train \
        /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5/split/tune5-eval5/Glaucoma_all_split.csv \
    --csv_file_test \
        /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5/split/tune5-eval5/Glaucoma_all_split.csv \
    --data_set \
        UF-Glaucoma \
    --uf_modality \
        bscan \
        slo
```

`--csv_file_train` and `--csv_file_test` point at the same per-task CSV
above — that's not a typo, it's the OphFoundation/UF-cohort convention: one
CSV per task (e.g. `Glaucoma_all_split.csv`, `DR_all_split.csv`,
`AMD_all_split.csv`, ...), split into train/val/test purely via its
`split` column. `--data_root` is the root directory the CSV's `folder`
column is relative to; `--data_set` is only a free-form label used for the
output directory name. Number of classes and the auto batch-size heuristic
are computed from the `label` column of the training split, instead of
counting image folders. See `run_uf.sh` for a ready-to-edit copy of this
command.


### True multi-modal (OCT B-scan + SLO together)

`run_cls_tuning_UF.py` picks one modality per run. `run_cls_tuning_UF_multimodaliy.py`
instead feeds MIRAGE both the middle OCT B-scan slice and the SLO scan of
each sample at once: `mutils/dataset_uf.py`'s `UFCohortMultiModalDataset`
returns `({'bscan': ..., 'slo': ...}, label)`, and `MIRAGEClsGlobal` (see
`mirage_wrapper.py`) concatenates both domains' tokens and lets the shared
encoder jointly attend to them before pooling — this only works because
MIRAGE's core architecture (MultiMAE) is inherently multi-modal; the
classification head previously just hard-asserted a single input domain.

Same required arguments as `run_cls_tuning_UF.py` minus `--uf_modality`
(always both domains):

### Subsampling the train split

Both UF scripts support `--new_subset_num` for data-efficiency/few-shot
sweeps: if set to a value > 0, the train split is subsampled down to that
many samples total, class-balanced proportionally to each class's share of
the full train split (every class keeps at least one sample; a class with
fewer samples than its computed share just keeps all of it). `--subsetseed`
(default `42`) seeds this sampling independently of `--seed`, so the same
subset can be reused across different training seeds. It's disabled by
default (`--new_subset_num 0`), which uses the full train split unchanged.
Val/test splits are never subsampled. See `mutils.misc.subsample_class_balanced`
for the implementation.

```bash
./runner python run_cls_tuning_UF_multimodaliy.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --linear_probing \
    --data_root \
        /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ \
    --csv_file_train \
        /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5/split/tune5-eval5/Glaucoma_all_split.csv \
    --csv_file_test \
        /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5/split/tune5-eval5/Glaucoma_all_split.csv \
    --data_set \
        UF-Glaucoma-mm
```

Same one-CSV-per-task convention as above. Results land in
`./__output/cls_uf_mm` by default, separate from both `run_cls_tuning.py`
(`./__output/cls`) and `run_cls_tuning_UF.py` (`./__output/cls_uf`). See
`run_uf_multimodal.sh` for a ready-to-edit copy of this command.


### Metrics and wandb logging

Both UF scripts use a different metric set than `run_cls_tuning.py`,
ported from OphFoundation's own `evaluate_model` (in
`pytorch_image_classification_our_model-v2-UF.py`) so numbers are directly
comparable to OphFoundation's results — see `mutils/metrics_uf.py`
(`compute_metrics_uf`, `safe_for_wandb`), used by both
`mutils/classification_uf.py` (single-modality) and
`mutils/classification_mm.py` (multi-modal). Two differences from
`run_cls_tuning.py`'s metrics (`mutils/classification.py`):

- **Macro averaging**, not weighted-by-class-support, for F1/AUROC/AP —
  every class counts equally regardless of how many samples it has.
- **Scaled to 0-100**, not 0-1.

It also adds precision, recall, Cohen's kappa, the confusion matrix, and
the classification report, none of which `run_cls_tuning.py` computes.
Plain accuracy replaces `run_cls_tuning.py`'s balanced accuracy (`bacc`) —
this is OphFoundation's convention, not a MIRAGE default, so
`--val_metric`'s default is `auroc` here (rather than `bacc`) to match.

Both scripts log to [Weights & Biases](https://wandb.ai) every epoch
(train/val metrics, including the confusion matrix and classification
report) and at the end (best-epoch validation summary, final test
metrics), mirroring OphFoundation's `wandb.log` calls. This needs
`wandb login` once per machine/account; if that isn't set up, pass
`--wandb_mode offline` (logs locally, sync later with `wandb sync`) or
`--wandb_mode disabled` (skip wandb, everything else — CSVs, checkpoints,
console output — is unaffected). `--wandb_project` sets the project name
(defaults: `MIRAGE_UF_result` / `MIRAGE_UF_result_mm`).

CSVs (`train_eval.csv`, `valid_eval.csv`, `test_eval.csv`) only hold the
scalar metrics — the confusion matrix and classification report aren't
flat values, so they're wandb/console-only, not CSV columns.


## Adding a new model

To add a new model, you need to create a new model class in `fm_cls_config.py` extending the `FoundModel` class in the same file.
You can check the existing adapted models in the file for reference.

