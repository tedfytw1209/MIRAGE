# Design: public-dataset classification scripts for fundus and OCT B-scan benchmarks

Date: 2026-07-20
Status: Approved

## Context

`run_cls_tuning.py` is MIRAGE's generic classification-tuning entry point. It
assumes each dataset is already laid out as pre-split
`train/val/test/Class_x/*.png` image folders (the "public dataset setting",
documented in `docs/classification_benchmark.md`) and loads each split with
plain `torchvision.datasets.ImageFolder`. It always feeds images through
MIRAGE's `bscan` input domain (the `MIRAGEFM` model wrapper defaults
`modalities = getattr(args, 'uf_modality', 'bscan')`, and `run_cls_tuning.py`
never sets that attribute).

The user wants two new entry points, modeled on reference scripts in the
sibling `OphFoundation` repo, for datasets not yet in MIRAGE:

- **Fundus** (`slm/finetune/2025-1212-finetune-publicbench-fundus-params.sh`):
  7 public color fundus photography datasets — `Glaucoma_fundus`,
  `IDRiD_data`, `JSIEC`, `MESSIDOR2`, `PAPILA`, `Retina`, `APTOS2019`.
- **OCT B-scan** (`slm/finetune/finetune-UF-benchmark_{duke14,glaucoma,oimhs,umn}_single.sh`):
  4 public OCT benchmark datasets sourced from OCTCubeM assets — `duke14`,
  `glaucoma` (OCT, distinct from MIRAGE's existing `Harvard_Glaucoma`),
  `oimhs`, `umn`.

These OphFoundation scripts run OphFoundation's own model/pipeline (CSV
fold-splits for OCT, `ImageFolder` for fundus) — they are references for
*which datasets and hyperparameters*, not for *how MIRAGE loads data*.
MIRAGE's own "public dataset setting" (pre-split folders) is what the new
scripts must use, per explicit instruction, not the CSV/fold convention.

## Decisions (confirmed with user)

1. **Data layout**: assume the 11 new datasets will be provided already
   split into `train/val/test/Class_x/` folders, exactly like every dataset
   `run_cls_tuning.py` already handles. No conversion/download code is
   written as part of this work.
2. **Two scripts**: build both `run_cls_tuning_fundus.py` and
   `run_cls_tuning_bscan.py`, as standalone, self-contained files — matching
   this repo's existing convention (`run_cls_tuning_UF.py` and
   `run_cls_tuning_UF_multimodaliy.py` are independent files, not a shared
   abstraction), rather than adding a `--modality` flag to
   `run_cls_tuning.py` itself.

## What each script does

Both are structural copies of `run_cls_tuning.py` (same argparse shape,
`process_args`, `build_dataset` via `ImageFolder`, `EarlyStopping`-driven
training loop, `train_eval.csv`/`valid_eval.csv`/`test_eval.csv` outputs,
weighted 0-1 metrics from `mutils/classification.py`). No wandb, no
macro/0-100 metrics — that's the `run_cls_tuning_UF*.py` lineage, not this
one.

The only functional delta in each is which MIRAGE input domain images are
routed through, set via `args.uf_modality` before `model_config` is built
(already read by `MIRAGEFM.__init__`; no changes needed to
`fm_cls_config.py` or `mirage_wrapper.py`):

- `run_cls_tuning_fundus.py` sets `args.uf_modality = 'slo'`. MIRAGE has no
  dedicated `fundus` domain (`mirage_wrapper.DOMAIN_CONF` only has `bscan`,
  `slo`, `bscanlayermap`), so color fundus photos route through `slo`, the
  only other en-face 2D domain. The shared transform's
  `Grayscale(num_output_channels=1)` step is mandatory, not a stylistic
  choice — MIRAGE's `slo` input adapter is single-channel by architecture
  (`DEFAULT_CONF['channels'] = 1`) — so this is a genuine cross-domain
  transfer evaluation, not a full-fidelity color-fundus setup. Default
  `--base_output_dir`: `./__output/cls_fundus`.
- `run_cls_tuning_bscan.py` sets `args.uf_modality = 'bscan'` explicitly
  (functionally identical to `run_cls_tuning.py`'s implicit default, but
  spelled out so the script is self-documenting and doesn't silently
  inherit behavior from `FoundModel`'s default). Default
  `--base_output_dir`: `./__output/cls_bscan`.

Both keep `run_cls_tuning.py`'s automatic `num_classes` inference (counting
class subfolders under `train/`) rather than a hardcoded per-dataset
class-count table — this is already dataset-agnostic and more robust than
copying the OphFoundation scripts' hardcoded `(dataset, num_class)` lists.

**Explicit non-goal**: `run_cls_tuning_bscan.py`'s data-loading logic ends up
nearly identical to today's `run_cls_tuning.py` (which already defaults to
`bscan` + `ImageFolder`). Its value is being a clearly-named, dedicated
entry point for the new OCT datasets — not new loading logic. No
per-volume/slice-grouping logic is added; each image file under a class
folder is one independent sample, same as every other MIRAGE classification
dataset today. If the underlying data is full multi-slice volumes rather
than one-file-per-sample, picking a representative slice is the user's data
-prep step, done before these scripts run.

## Shell scripts

`run_fundus.sh` and `run_bscan.sh`, modeled directly on `run_uf.sh`:
- Single dataset per invocation via a positional arg (default placeholder,
  edit-before-run, same as `run_uf.sh`'s `TASK` variable).
- `./runner` fan-out over `--weights` (MIRAGE-Base and MIRAGE-Large), same
  as every other MIRAGE shell script.
- `--seed 0` only — no multi-seed loop (standing user preference; see
  `run_uf_all_tasks.sh`/`run_uf.sh` history: multi-seed sweeps are added
  only when explicitly requested, after a staged single-seed validation
  run).
- `--linear_probing` passed through as an optional positional flag, same
  pattern as `run_uf.sh`'s `LINEAR_PROBING`.
- A comment block listing the known dataset names (and, for fundus, the
  OphFoundation reference's per-dataset class counts, for documentation
  only — the scripts themselves auto-infer `num_classes`).

## Explicitly out of scope

- No dataset download/conversion/splitting code (no changes to
  `prepare_env.py`).
- No wandb integration.
- No "all-tasks" driver script (like `run_uf_all_tasks.sh`) looping over
  all 7 fundus / 4 OCT datasets — can be added later on request.
- No multi-seed sweeping.
- No hardcoded per-dataset num_classes table used by the code (documentation
  comment only).

## Docs

Add a short section to `docs/classification_tuning.md` documenting both new
scripts (mirroring how `run_cls_tuning_UF.py` is documented there), so the
"which script for which dataset layout" picture stays complete in one place.

## Files touched

- New: `run_cls_tuning_fundus.py`
- New: `run_cls_tuning_bscan.py`
- New: `run_fundus.sh`
- New: `run_bscan.sh`
- Edit: `docs/classification_tuning.md`
