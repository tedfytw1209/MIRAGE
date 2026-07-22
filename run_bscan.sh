
source ./venv/bin/activate


# Public OCT B-scan classification (MIRAGE's default 'bscan' domain -- see
#   run_cls_tuning_bscan.py; a dedicated entry point for these datasets,
#   functionally the same pipeline as run_cls_tuning.py).
#
# Datasets follow MIRAGE's "public dataset setting": pre-split
#   train/val/test/Class_x/ image folders under --data_root (see
#   docs/classification_benchmark.md). Number of classes is auto-inferred
#   from the folder structure.
#
# Known datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark against these same public OCT
#   datasets via their own CSV/fold pipeline): duke14, glaucoma, oimhs, umn
DATASET=${1:-"duke14"}
LINEAR_PROBING=${2:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public OCT B-scan datasets (see docs/classification_benchmark.md).
#   IMPORTANT: OphFoundation's own raw data is volumetric (one CSV row per
#   volume, resampled to 20 slices at load time -- see the caveat in
#   run_cls_tuning_bscan.py's module docstring). Each Class_x/ file here
#   must be ONE selected slice per volume, not every raw slice PNG.
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_bscan_public/"

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

./runner python run_cls_tuning_bscan.py \
    --runners 1 \
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
        $DATASET \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan
exit
