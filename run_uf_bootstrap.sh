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


# Bootstrap sweep: resample the (class-balanced) --new_subset_num training
#   subset with N different --subsetseed values to get a distribution of
#   results for a fixed subset size, instead of a single point estimate --
#   mirrors OphFoundation's bootstrap convention (--bootstrap_runs /
#   --new_subset_num / --subsetseed swept over a SUBSET_SEEDS list).
#
# Scope: run_cls_tuning_UF_multimodaliy.py (true multi-modal), MIRAGE-Large,
#   --linear_probing. See run_uf_bootstrap_base.sh for the MIRAGE-Base +
#   full-fine-tune counterpart; single-modality bootstrap
#   (run_cls_tuning_UF.py) is still to do.
DATA_TYPE="IRB2024_v5"
SUBSET_NUM=${1:-500}
DATA_ROOT="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"

# 18 tasks (number of classes noted for reference only -- each script
#   auto-detects num_classes from its CSV's `label` column at runtime):
#   AMD:2 Cataract:2 DR:6 Glaucoma:6 DR_binary:2 Glaucoma_binary:2
#   DME:5 CSR:2 Drusen:2 ERM:2 MH:2 CRVO_CRAO:2 PVD:2 RNV:2 DME_binary:2
#   PD:2 DKD:2 Diabetes:2
TASKS=(
    AMD Cataract DR Glaucoma DR_binary Glaucoma_binary
    DME CSR Drusen ERM MH CRVO_CRAO PVD RNV DME_binary
    PD DKD Diabetes
)

for TASK in "${TASKS[@]}"; do
    UF_CSV="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${DATA_TYPE}/split/tune5-eval5/${TASK}_all_split.csv"
    echo "=== Task: ${TASK} ==="

    # 10 bootstrap resamples of the training subset (same subset size, 10
    #   different random draws) -- matches the reference scripts'
    #   SUBSET_SEEDS=(1 2 ... 10). --runners 10 runs all of them
    #   concurrently within a task (subsetted + linear-probing is cheap
    #   on GPU memory); tasks themselves are looped sequentially.
    #
    # --wandb_tags keeps the tag list down to the axes worth filtering on
    #   ('bootstrap' + 'sub100'/'sub300'/'sub500'); without it the tag is the
    #   per-task --data_set (UF-Diabetes-bootstrap-sub100, ...), i.e. 18 tasks
    #   x 3 sizes = 54 single-use tags. The task stays in the run name and in
    #   the logged config either way.
    ./runner python run_cls_tuning_UF_multimodaliy.py \
        --runners 10 \
        -- \
        --version v1 \
        --seed 0 \
        --weights \
            $WEIGHTS_LARGE \
        --linear_probing \
        --data_root \
            $DATA_ROOT \
        --csv_file_train \
            $UF_CSV \
        --csv_file_test \
            $UF_CSV \
        --data_set \
            UF-${TASK}-bootstrap-sub${SUBSET_NUM} \
        --base_output_dir \
            /blue/ruogu.fang/tienyuchang/MIRAGE_results/cls_uf_mm_bootstrap \
        --new_subset_num \
            $SUBSET_NUM \
        --subsetseed \
            1 \
            2 \
            3 \
            4 \
            5 \
            6 \
            7 \
            8 \
            9 \
            10 \
        --wandb_project \
            MIRAGE_UF_result \
        --wandb_tags \
            bootstrap,sub${SUBSET_NUM} \
        --wandb_mode \
            online

    echo "=== Task ${TASK} done ==="
done
exit
