#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpg-b200
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

source ./venv/bin/activate

# Regenerates every checkpoint run_uf_all_tasks.sh trains (bscan-only,
#   slo-only, and joint multimodal) -- e.g. after deleting them -- and, once
#   retraining finishes, runs the dual-modality late-fusion sweep
#   (run_uf_dual_modality_latefusion.sh) on the fresh checkpoints.
#
# WANDB_TAGS=rerun makes run_uf_all_tasks.sh tag these wandb runs "rerun"
#   (replacing the per-task tag, same convention as run_uf_bootstrap*.sh --
#   the task stays visible in the run name/config either way) so they're
#   distinguishable in the dashboard from any previous run of the same task.
#
# Each stage is invoked as a subprocess (not sourced), so its trailing
#   `exit` only ends that stage, not this wrapper.
export WANDB_TAGS="rerun"
bash run_uf_all_tasks.sh

bash run_uf_dual_modality_latefusion.sh

exit
