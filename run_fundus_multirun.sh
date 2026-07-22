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


# Public fundus-photo classification -- all 7 datasets in a single
#   ./runner call (2 weights x 7 datasets = 14 combinations, run via
#   --runners 14), instead of looping over datasets in bash like
#   run_fundus_all_tasks.sh. Only one --linear_probing mode per invocation:
#   ./runner fans out multi-value flags, but can't fan out a flag's
#   presence/absence, so both modes in one script still needs the bash
#   loop (run_fundus_all_tasks.sh).
LINEAR_PROBING=${1:-true}  # true: freeze encoder (linear probe); false: full fine-tune

# EDIT ME: root containing pre-split train/val/test/Class_x/ folders for
#   these public fundus datasets (see docs/classification_benchmark.md).
DATA_ROOT="/orange/ruogu.fang/tienyuchang/MIRAGE_data/cls_fundus_public/"

LINEAR_PROBING_FLAG=""
if [ "$LINEAR_PROBING" = "true" ]; then
    LINEAR_PROBING_FLAG="--linear_probing"
fi

# 7 datasets (name: num_class, per OphFoundation's
#   2025-1212-finetune-publicbench-fundus-params.sh reference). num_classes
#   is auto-inferred by run_cls_tuning_fundus.py from the folder structure,
#   not passed here -- listed for reference only.
#   Glaucoma_fundus:3 IDRiD_data:5 JSIEC:39 MESSIDOR2:5 PAPILA:3 Retina:4 APTOS2019:5
./runner python run_cls_tuning_fundus.py \
    --runners 14 \
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
        Glaucoma_fundus \
        IDRiD_data \
        JSIEC \
        MESSIDOR2 \
        PAPILA \
        Retina \
        APTOS2019 \
    --base_output_dir \
        /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_fundus
exit
