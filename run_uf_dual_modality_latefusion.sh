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

DATA_TYPE="IRB2024_v5"
DATA_ROOT="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"
CLS_UF_DIR="/orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_uf"
LATEFUSION_DIR="/orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_uf_latefusion"

# 18 tasks (number of classes noted for reference only -- each script
#   auto-detects num_classes from its CSV's `label` column at runtime):
#   AMD:2 Cataract:2 DR:6 Glaucoma:6 DR_binary:2 Glaucoma_binary:2
#   DME:5 CSR:2 Drusen:2 ERM:2 MH:2 CRVO_CRAO:2 PVD:2 RNV:2 DME_binary:2
#   PD:2 DKD:2 Diabetes:2
# Keep in sync with run_uf_all_tasks.sh's TASKS -- only trained checkpoints
#   have predictions to fuse.
TASKS=(
    AMD Cataract DR Glaucoma DR_binary Glaucoma_binary
    DME CSR Drusen ERM MH CRVO_CRAO PVD RNV DME_binary
    PD DKD Diabetes
    Glaucoma_fbinary Glaucoma_filtered DR_fbinary DR_filtered
)


# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: TASK
# $3: UF_CSV
regenerate_predictions() {
    local PROBE_FLAG=$1
    local TASK=$2
    local UF_CSV=$3
    ./runner python run_cls_tuning_UF.py \
        --runners 1 \
        -- \
        --version v1 \
        --seed 0 \
        --weights \
            $WEIGHTS_BASE \
            $WEIGHTS_LARGE \
        $PROBE_FLAG \
        --data_root \
            $DATA_ROOT \
        --csv_file_train \
            $UF_CSV \
        --csv_file_test \
            $UF_CSV \
        --data_set \
            UF-${TASK} \
        --base_output_dir \
            $CLS_UF_DIR \
        --uf_modality \
            bscan \
            slo \
        --eval \
        --save_predictions \
        --wandb_mode \
            disabled
}

DATASETS=()
for TASK in "${TASKS[@]}"; do
    UF_CSV="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${DATA_TYPE}/split/tune5-eval5/${TASK}_all_split.csv"
    echo "=== Task: ${TASK} ==="

    for PROBE_FLAG in "" "--linear_probing"; do
        regenerate_predictions "$PROBE_FLAG" "$TASK" "$UF_CSV"
    done

    DATASETS+=("UF-${TASK}")
    echo "=== Task ${TASK} predictions regenerated ==="
done

python late_fusion_uf.py \
    --base_output_dir $CLS_UF_DIR \
    --out_base_output_dir $LATEFUSION_DIR \
    --version v1 \
    --seed 0 \
    --datasets "${DATASETS[@]}" \
    --model_names mirage-base mirage-large \
    --probe_tags finetune linear \
    --wandb_project MIRAGE_UF_result

exit
