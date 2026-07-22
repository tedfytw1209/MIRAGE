#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

source ./venv/bin/activate

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public fundus datasets (see docs/classification_benchmark.md).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_fundus_public/"
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
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_fundus &
}

for DATASET in "${DATASETS[@]}"; do
    echo "=== Dataset: ${DATASET} ==="
    PIDS=()
    for PROBE_FLAG in "" "--linear_probing"; do
        launch "$PROBE_FLAG" "$DATASET"
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
    echo "=== Dataset ${DATASET} done ==="
done
exit
