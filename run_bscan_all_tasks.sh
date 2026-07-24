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

# Root containing the raw per-dataset OCT volume/slice image folders (one
#   subfolder per class, e.g. duke14_processed/{AMD,DME,NORMAL}) --
#   see mutils/dataset_public_oct.py for the exact per-dataset subdirectory
#   layout and CSV/fold-split convention. --fold 0 uses the first
#   pre-computed train/val/test partition; looping over multiple folds is
#   not done here (staged rollout -- add a fold loop only once fold 0 is
#   validated).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"
CSV_ROOT="/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"
FOLD=0
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from each dataset's fold-split CSV, not passed
#   here.
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
        --csv_root \
            $CSV_ROOT \
        --fold \
            $FOLD \
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

