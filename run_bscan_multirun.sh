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


# Public OCT B-scan classification -- all 4 datasets in a single ./runner
#   call (2 weights x 4 datasets = 8 combinations, run via --runners 8),
#   instead of looping over datasets in bash like run_bscan_all_tasks.sh.
#   Only one --linear_probing mode per invocation: ./runner fans out
#   multi-value flags, but can't fan out a flag's presence/absence, so
#   both modes in one script still needs the bash loop (run_bscan_all_tasks.sh).
#
# Datasets follow the CSV/fold-split convention (see
#   mutils/dataset_public_oct.py): raw per-slice volume folders under
#   --data_root, plus a fold-split CSV under --csv_root that assigns each
#   volume to train/val/test and gives its integer label. --fold 0 uses the
#   first pre-computed partition (see run_cls_tuning_bscan.py's module
#   docstring for why looping over folds isn't done automatically).
LINEAR_PROBING=${1:-false}  # true: freeze encoder (linear probe); false: full fine-tune

# Root containing the raw per-dataset OCT volume/slice image folders.
DATA_ROOT="/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"
CSV_ROOT="/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"
FOLD=0

# Array, not a string: the fine-tune case must expand to zero words, not
#   one empty argument. runner's value lookahead would otherwise fold ""
#   into the --weights list and emit bogus extra combinations.
LINEAR_PROBING_FLAG=()
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG=(--linear_probing)
fi

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from each dataset's fold-split CSV, not passed
#   here.
./runner python run_cls_tuning_bscan.py \
    --runners 8 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth \
    "${LINEAR_PROBING_FLAG[@]}" \
    --data_root \
        "$DATA_ROOT" \
    --csv_root \
        "$CSV_ROOT" \
    --fold \
        "$FOLD" \
    --data_set \
        duke14 \
        glaucoma \
        oimhs \
        umn \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan
exit
