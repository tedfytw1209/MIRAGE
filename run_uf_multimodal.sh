
source ./venv/bin/activate


# UF cohort classification, true multi-modal (OCT B-scan + SLO fed to
#   MIRAGE together, jointly attended to by the shared encoder)
#
# Paths follow the OphFoundation/UF-cohort convention on HiPerGator's
#   "orange" storage: --csv_file_train and --csv_file_test point at the
#   SAME per-task CSV, which is split into train/val/test via its 'split'
#   column (see slm/finetune/finetune-UF-benchmark_IRB2024v5_single.sh in
#   OphFoundation).
DATA_TYPE="IRB2024_v5"
TASK=${1:-"Glaucoma"}  # AMD, Cataract, DR, Glaucoma, DR_binary, Glaucoma_binary
DATA_ROOT="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
UF_CSV="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${DATA_TYPE}/split/tune5-eval5/${TASK}_all_split.csv"

# Metrics/wandb logging follow OphFoundation's own evaluate_model/wandb
#   convention (macro-averaged, 0-100 scaled; see mutils/metrics_uf.py).
# Requires `wandb login` once per machine/account; use
#   --wandb_mode offline (logs locally, sync later with `wandb sync`) or
#   --wandb_mode disabled (skip wandb entirely) if that isn't set up.
./runner python run_cls_tuning_UF_multimodaliy.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Base.pth \
        /orange/ruogu.fang/tienyuchang/MIRAGE_pretrain/MIRAGE-Large.pth \
    --linear_probing \
    --data_root \
        $DATA_ROOT \
    --csv_file_train \
        $UF_CSV \
    --csv_file_test \
        $UF_CSV \
    --data_set \
        UF-${TASK}-mm \
    --wandb_project \
        MIRAGE_UF_result_mm \
    --wandb_mode \
        online
exit
