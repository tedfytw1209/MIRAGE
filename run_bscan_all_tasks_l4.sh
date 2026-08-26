#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=144:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

source ./venv/bin/activate

# L4 variant: the L4 only has 14GB of VRAM, so unlike
#   run_bscan_all_tasks.sh (B200, 2 probe modes x 2 model sizes run
#   concurrently) everything here runs strictly one process at a time --
#   one probe mode, one model size, one fold in the GPU at any moment.
# This script handles exactly ONE dataset per submission (passed as $1),
#   looping over that dataset's 10 folds inside the job. Don't sbatch this
#   directly -- use submit_bscan_l4.sh, which sbatch-submits one instance
#   of this script per dataset (no job array, since arrays make it harder
#   to get scheduled -- separate independent jobs queue more easily).
DATASET=$1

# Root containing the raw per-dataset OCT volume/slice image folders (one
#   subfolder per class, e.g. duke14_processed/{AMD,DME,NORMAL}) --
#   see mutils/dataset_public_oct.py for the exact per-dataset subdirectory
#   layout and CSV/fold-split convention. Loops over all 10 pre-computed
#   fold-split partitions (0-9) so summarize_bscan_results.py can report
#   mean +/- std across folds, not just a fold-0 point estimate.
DATA_ROOT="/blue/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"
CSV_ROOT="/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"
FOLDS=(0 1 2 3 4 5 6 7 8 9)
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"
BASE_OUTPUT_DIR="/orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan"

# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: DATASET
# $3: FOLD
launch() {
    local PROBE_FLAG=$1
    local DATASET=$2
    local FOLD=$3
    ./runner python run_cls_tuning_bscan.py \
        --runners 1 \
        -- \
        --version v1 \
        --seed 0 \
        --weights \
            $WEIGHTS_BASE \
        $PROBE_FLAG \
        --data_root \
            $DATA_ROOT \
        --csv_root \
            $CSV_ROOT \
        --fold \
            $FOLD \
        --data_set \
            $DATASET \
        --base_output_dir \
            $BASE_OUTPUT_DIR
}

echo "=== Dataset: ${DATASET} ==="
for FOLD in "${FOLDS[@]}"; do
    echo "--- Fold: ${FOLD} ---"
    launch "--linear_probing" "$DATASET" "$FOLD"
    echo "--- Fold ${FOLD} done ---"
done
echo "=== Dataset ${DATASET} done ==="
