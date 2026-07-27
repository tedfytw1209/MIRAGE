#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

source ./venv/bin/activate

# L4 variant: the L4 only has 14GB of VRAM, so unlike
#   run_fundus_all_tasks.sh (B200, 2 probe modes x 2 model sizes run
#   concurrently) everything here runs strictly one process at a time --
#   one probe mode, one model size, one dataset in the GPU at any moment.

# Root containing pre-split train/val/test/Class_x/ folders for these
#   public fundus datasets, per OphFoundation's reference benchmark script
#   (dataset_dir/dataset_name/{train,val,test}/Class_x/).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/"
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"

# 7 datasets (name: num_class, per OphFoundation's
#   2025-1212-finetune-publicbench-fundus-params.sh reference). num_classes
#   is auto-inferred by run_cls_tuning_fundus.py from the folder structure,
#   not passed here -- listed for reference only.
#   Glaucoma_fundus:3 IDRiD_data:5 JSIEC:39 MESSIDOR2:5 PAPILA:3 Retina:4 APTOS2019:5
DATASETS=(Glaucoma_fundus IDRiD_data JSIEC MESSIDOR2 PAPILA Retina APTOS2019)

# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: DATASET
launch() {
    local PROBE_FLAG=$1
    local DATASET=$2
    ./runner python run_cls_tuning_fundus.py \
        --runners 1 \
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
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_fundus
}

for DATASET in "${DATASETS[@]}"; do
    echo "=== Dataset: ${DATASET} ==="
    launch "--linear_probing" "$DATASET"
    echo "=== Dataset ${DATASET} done ==="
done
