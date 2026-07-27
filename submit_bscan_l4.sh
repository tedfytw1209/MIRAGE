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

# Usage: ./submit_bscan_l4.sh

# 4 datasets (per OphFoundation's finetune-UF-benchmark_*_single.sh
#   reference scripts, which benchmark these same public OCT datasets via
#   their own CSV/fold pipeline). num_classes is auto-inferred by
#   run_cls_tuning_bscan.py from the folder structure, not passed here.
DATASETS=(duke14 glaucoma oimhs umn)

for DATASET in "${DATASETS[@]}"; do
    echo "Submitting job for dataset: ${DATASET}"
    sbatch --job-name="bscan_${DATASET}" run_bscan_all_tasks_l4.sh "$DATASET"
done
