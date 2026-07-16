
source ./venv/bin/activate


# Sweep across all 18 UF-cohort tasks x every modality configuration
#   (bscan, slo, true multi-modal) x both MIRAGE sizes (Base, Large) x
#   both probing modes (full fine-tune and --linear_probing, encoder
#   frozen -- the paper's own setup). run_uf.sh / run_uf_multimodal.sh
#   cover the same two scripts for a single task/probing-mode at a time;
#   this one runs everything.
#
# A single run only uses ~3% of GPU memory, so per task all 4 `./runner`
#   calls below (fine-tune-single, fine-tune-multimodal,
#   linear-probe-single, linear-probe-multimodal) are launched
#   concurrently: single-modality ones fan out over --uf_modality
#   bscan/slo x --weights Base/Large (4 combinations, `--runners 4`),
#   multi-modal ones over --weights Base/Large (2 combinations,
#   `--runners 2`). That's up to 4+2+4+2 = 12 concurrent processes per
#   task (~36% of GPU memory), backgrounded together and waited on
#   before moving to the next task. Tasks themselves are looped
#   sequentially -- lower/raise the --runners counts, or drop the `wait`
#   between tasks, to trade off cross-task concurrency vs. memory
#   headroom on your node.
DATA_TYPE="IRB2024_v5"
DATA_ROOT="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
WEIGHTS_BASE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth"
WEIGHTS_LARGE="/orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth"

# 18 tasks (number of classes noted for reference only -- each script
#   auto-detects num_classes from its CSV's `label` column at runtime):
#   AMD:2 Cataract:2 DR:6 Glaucoma:6 DR_binary:2 Glaucoma_binary:2
#   DME:5 CSR:2 Drusen:2 ERM:2 MH:2 CRVO_CRAO:2 PVD:2 RNV:2 DME_binary:2
#   PD:2 DKD:2 Diabetes:2
#TASKS=(
#    AMD Cataract DR Glaucoma DR_binary Glaucoma_binary
#    DME CSR Drusen ERM MH CRVO_CRAO PVD RNV DME_binary
#    PD DKD Diabetes
#)
TASKS=(
    CRVO_CRAO PVD RNV DME_binary
    PD DKD Diabetes
)

# $1: PROBE_FLAG ("" for full fine-tune, "--linear_probing" otherwise)
# $2: TASK
# $3: UF_CSV
launch_single_modality() {
    local PROBE_FLAG=$1
    local TASK=$2
    local UF_CSV=$3
    ./runner python run_cls_tuning_UF.py \
        --runners 4 \
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
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_uf \
        --uf_modality \
            bscan \
            slo \
        --wandb_project \
            MIRAGE_UF_result \
        --wandb_mode \
            online &
}

launch_multimodal() {
    local PROBE_FLAG=$1
    local TASK=$2
    local UF_CSV=$3
    ./runner python run_cls_tuning_UF_multimodaliy.py \
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
        --csv_file_train \
            $UF_CSV \
        --csv_file_test \
            $UF_CSV \
        --data_set \
            UF-${TASK}-mm \
        --base_output_dir \
            /orange/ruogu.fang/tienyuchang/MIRAGE_results/cls_uf_mm \
        --wandb_project \
            MIRAGE_UF_result \
        --wandb_mode \
            online &
}

for TASK in "${TASKS[@]}"; do
    UF_CSV="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${DATA_TYPE}/split/tune5-eval5/${TASK}_all_split.csv"
    echo "=== Task: ${TASK} ==="

    PIDS=()
    for PROBE_FLAG in "" "--linear_probing"; do
        launch_single_modality "$PROBE_FLAG" "$TASK" "$UF_CSV"
        PIDS+=($!)
        launch_multimodal "$PROBE_FLAG" "$TASK" "$UF_CSV"
        PIDS+=($!)
    done

    wait "${PIDS[@]}"
    echo "=== Task ${TASK} done ==="
done
exit
