#!/usr/bin/env bash
#SBATCH --job-name=antmaze-dmlp-full
#SBATCH --output=repro_runs/slurm-%x-%j.out
#SBATCH --error=repro_runs/slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

MODE="${1:-both}"
ROOT="/bigwork/nhwptarj/Johny-Thesis/external/stitching-is-combinatorial-generalisation"
VENV="/bigwork/nhwptarj/Johny-Thesis/venv"

cd "$ROOT"
source "$VENV/bin/activate"

mkdir -p repro_runs data

RUN_DIR="$ROOT/repro_runs/antmaze_umaze_dmlp_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > repro_runs/latest_antmaze_full_run_dir.txt

if [[ ! -f data/antmaze-umaze-v0.pkl ]]; then
  if [[ -f downloads/antmaze-umaze-v0.pkl ]]; then
    cp downloads/antmaze-umaze-v0.pkl data/antmaze-umaze-v0.pkl
  else
    echo "Missing data/antmaze-umaze-v0.pkl and downloads/antmaze-umaze-v0.pkl" >&2
    exit 1
  fi
fi

{
  echo "run_dir: $RUN_DIR"
  echo "mode: $MODE"
  echo "host: $(hostname)"
  echo "date: $(date --iso-8601=seconds)"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY
} | tee "$RUN_DIR/slurm_context_${SLURM_JOB_ID:-manual}.log"

python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
git rev-parse HEAD > "$RUN_DIR/submodule_commit.txt"
git diff > "$RUN_DIR/local_patch.diff"
cp cfgs/dmlp.yaml "$RUN_DIR/dmlp_original.yaml"

run_base() {
  PYTHONUNBUFFERED=1 python -u train_dmlp.py \
    dataset_name=antmaze-umaze-v0 \
    augment_data=False \
    augment_prob=0 \
    nclusters=20 \
    max_train_iters=25 \
    num_updates_per_iter=40000 \
    num_eval_ep=10 \
    num_workers=0 \
    save_snapshot=False \
    wandb_log=False \
    device=cuda \
    log_dir="$RUN_DIR/hydra_base_full" \
    2>&1 | tee "$RUN_DIR/train_eval_base_full.log"
}

run_aug() {
  PYTHONUNBUFFERED=1 python -u train_dmlp.py \
    dataset_name=antmaze-umaze-v0 \
    augment_data=True \
    augment_prob=0.5 \
    nclusters=20 \
    max_train_iters=25 \
    num_updates_per_iter=40000 \
    num_eval_ep=10 \
    num_workers=0 \
    save_snapshot=False \
    wandb_log=False \
    device=cuda \
    log_dir="$RUN_DIR/hydra_aug_full" \
    2>&1 | tee "$RUN_DIR/train_eval_aug_full.log"
}

case "$MODE" in
  base)
    run_base
    ;;
  aug)
    run_aug
    ;;
  both)
    run_base
    run_aug
    ;;
  *)
    echo "Usage: sbatch scripts/slurm_antmaze_dmlp_full.sbatch [base|aug|both]" >&2
    exit 2
    ;;
esac

grep "eval/avg_reward" "$RUN_DIR"/train_eval_*.log > "$RUN_DIR/eval_summary.txt" || true
echo "Done. Results in $RUN_DIR"
