#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
# Full fine-tuning is far slower per run than the linear-probing sweep in
#   run_uf_bootstrap.sh (all encoder weights get gradients, and only 2 runs
#   fit concurrently instead of 10), so the wall time is raised accordingly:
#   18 tasks x 10 bootstrap seeds at --runners 2.
#SBATCH --time=72:00:00
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
# Smaller-model / full-fine-tune counterpart of run_uf_bootstrap.sh (which
#   is MIRAGE-Large + --linear_probing). MIRAGE ships in two sizes only,
#   Base (ViT-B) and Large (ViT-L) -- see fm_cls_config.py's
#   `mirage-base`/`mirage-large` configs -- so "small" here is MIRAGE-Base.
#   Both scripts write under the same --base_output_dir: the per-run path
#   already encodes model name and `_finetune`/`_linear`, so results do not
#   collide and can be aggregated together.
DATA_TYPE="IRB2024_v5"
SUBSET_NUM=${1:-500}
DATA_ROOT="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"

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
    #   SUBSET_SEEDS=(1 2 ... 10). Only --runners 2 here (vs 10 for linear
    #   probing): full fine-tuning keeps optimizer state and activations for
    #   the whole encoder, so more concurrent runs risk OOM -- same limit
    #   run_uf_all_tasks.sh uses for the multi-modal script.
    #
    # --wandb_tags keeps the tag list down to the axes worth filtering on
    #   ('bootstrap' + 'sub100'/'sub300'/'sub500'); without it the tag is the
    #   per-task --data_set (UF-Diabetes-bootstrap-sub100, ...), i.e. 18 tasks
    #   x 3 sizes = 54 single-use tags. The task stays in the run name and in
    #   the logged config either way; 'mirage-base' and 'finetune' tags are
    #   added by the script itself.
    ./runner python run_cls_tuning_UF_multimodaliy.py \
        --runners 2 \
        -- \
        --version v1 \
        --seed 0 \
        --weights \
            $WEIGHTS_BASE \
        --data_root \
            $DATA_ROOT \
        --csv_file_train \
            $UF_CSV \
        --csv_file_test \
            $UF_CSV \
        --data_set \
            UF-${TASK}-bootstrap-sub${SUBSET_NUM} \
        --base_output_dir \
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_uf_mm_bootstrap \
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
