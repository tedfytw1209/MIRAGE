#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

source ./venv/bin/activate

# Root containing the raw per-dataset OCT volume/slice image folders (one
#   subfolder per class, e.g. duke14_processed/{AMD,DME,NORMAL}) --
#   see mutils/dataset_public_oct.py for the exact per-dataset subdirectory
#   layout and CSV/fold-split convention. Loops over all 10 pre-computed
#   fold-split partitions (0-9) so summarize_bscan_results.py can report
#   mean +/- std across folds, not just a fold-0 point estimate.
DATA_ROOT="/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"
CSV_ROOT="/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"
BASE_OUTPUT_DIR="/orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan"

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark these same public OCT datasets via
#   their own CSV/fold pipeline). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from the folder structure, not passed here.
#DATASETS=(duke14 glaucoma oimhs umn)
DATASETS=(glaucoma)

# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: DATASET
# $3: FOLD
launch() {
    local PROBE_FLAG=$1
    local DATASET=$2
    local FOLD=$3
    ./runner python run_cls_tuning_bscan.py \
        --runners 2 \
        -- \
        --version v1 \
        --seed 0 \
        --weights \
            $WEIGHTS_LARGE \
        --data_root \
            $DATA_ROOT \
        --csv_root \
            $CSV_ROOT \
        --fold \
            $FOLD \
        --data_set \
            $DATASET \
        --base_output_dir \
            $BASE_OUTPUT_DIR &
}

for DATASET in "${DATASETS[@]}"; do
    echo "=== Dataset: ${DATASET} ==="
    for FOLD in "${FOLDS[@]}"; do
        echo "--- Fold: ${FOLD} ---"
        PIDS=()
        for PROBE_FLAG in "" "--linear_probing"; do
            launch "$PROBE_FLAG" "$DATASET" "$FOLD"
            PIDS+=($!)
        done
        wait "${PIDS[@]}"
        echo "--- Fold ${FOLD} done ---"
    done
    echo "=== Dataset ${DATASET} done ==="
done
