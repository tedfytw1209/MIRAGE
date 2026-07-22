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
# Datasets follow MIRAGE's "public dataset setting": pre-split
#   train/val/test/Class_x/ image folders under --data_root (see
#   docs/classification_benchmark.md). IMPORTANT: OphFoundation's own raw
#   data is volumetric (one CSV row per volume, resampled to 20 slices at
#   load time -- see the caveat in run_cls_tuning_bscan.py's module
#   docstring). Each Class_x/ file here must be ONE selected slice per
#   volume, not every raw slice PNG.
LINEAR_PROBING=${1:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public OCT B-scan datasets (see docs/classification_benchmark.md).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_bscan_public/"

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark these same public OCT datasets via
#   their own CSV/fold pipeline). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from the folder structure, not passed here.
./runner python run_cls_tuning_bscan.py \
    --runners 8 \
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
        duke14 \
        glaucoma \
        oimhs \
        umn \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_bscan
exit
