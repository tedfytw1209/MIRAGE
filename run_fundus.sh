
source ./venv/bin/activate


# Public fundus-photo classification (color fundus photos routed through
#   MIRAGE's 'slo' domain -- see run_cls_tuning_fundus.py; MIRAGE has no
#   dedicated 'fundus' domain).
#
# Datasets follow MIRAGE's "public dataset setting": pre-split
#   train/val/test/Class_x/ image folders under --data_root (see
#   docs/classification_benchmark.md). Number of classes is auto-inferred
#   from the folder structure; the reference class counts below (from
#   OphFoundation's 2025-1212-finetune-publicbench-fundus-params.sh) are
#   for documentation only.
#
# Known datasets (name: num_class, per OphFoundation reference):
#   Glaucoma_fundus:3 IDRiD_data:5 JSIEC:39 MESSIDOR2:5 PAPILA:3 Retina:4 APTOS2019:5
DATASET=${1:-"Glaucoma_fundus"}
LINEAR_PROBING=${2:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public fundus datasets (see docs/classification_benchmark.md).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_fundus_public/"

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

./runner python run_cls_tuning_fundus.py \
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
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_fundus
exit
