#!/bin/bash
#SBATCH --job-name=judge_ft
#SBATCH --array=0-14          # 5 losses x 3 seeds
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_p2
#SBATCH --output=logs/judge_ft_%A_%a.out

# IDRIS A100 80GB, conda env: torch-multimodal (Python 3.12)
module purge
source activate torch-multimodal

LOSSES=(mse bradley_terry ranknet approxndcg listnet)
SEEDS=(42 1 2)
LOSS=${LOSSES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

echo "judge fine-tune: loss=$LOSS seed=$SEED"
python scripts/finetune_judge.py \
    --judge_model meta-llama/Llama-3.1-8B \
    --loss $LOSS --seed $SEED \
    --train_data esnli_human \
    --out results/judge_ft/${LOSS}_seed${SEED}/result.json
