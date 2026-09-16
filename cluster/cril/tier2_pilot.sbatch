#!/bin/bash
#SBATCH --job-name=arr-pythia-pilot
#SBATCH --partition=h100
#SBATCH --account=cril
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --gpu-bind=closest
#SBATCH --time=00:30:00
#SBATCH --array=0-5%2
#SBATCH --export=NIL
#SBATCH --output=runs/tier2_pythia/slurm/%x-%A_%a.out
#SBATCH --error=runs/tier2_pythia/slurm/%x-%A_%a.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/cluster/cril/common.sh"
arr_preamble

LOSSES=(listnet listmle mse)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
if (( TASK_ID < 3 )); then
  CONSTRUCTION=independent
  LOSS="${LOSSES[$TASK_ID]}"
else
  CONSTRUCTION=shared
  LOSS="${LOSSES[$((TASK_ID - 3))]}"
fi
OUTPUT="$ARR_PROJECT_ROOT/runs/tier2_pythia/pilots/${CONSTRUCTION}_${LOSS}"

if [[ "$CONSTRUCTION" == independent ]]; then
  arr_run -m src.arr.tier2_cli train-member \
    --config configs/arr/tier2_pythia.yaml \
    --loss "$LOSS" --global-seed 42 --member-id 0 \
    --arm independent --pilot --output "$OUTPUT"
else
  arr_run -m src.arr.tier2_cli train-shared \
    --config configs/arr/tier2_pythia.yaml \
    --loss "$LOSS" --global-seed 42 \
    --arm baseline --pilot --output "$OUTPUT"
fi
arr_run -m src.arr.tier2_cli verify-run --run "$OUTPUT"
