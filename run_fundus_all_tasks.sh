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

DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_bscan_public/"
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark these same public OCT datasets via
#   their own CSV/fold pipeline). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from the folder structure, not passed here.
DATASETS=(duke14 glaucoma oimhs umn)

# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: DATASET
launch() {
    local PROBE_FLAG=$1
    local DATASET=$2
    ./runner python run_cls_tuning_bscan.py \
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
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan &
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
