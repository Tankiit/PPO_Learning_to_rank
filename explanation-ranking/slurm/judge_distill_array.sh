#!/bin/bash
#SBATCH --job-name=judge_distill
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_p2
#SBATCH --output=logs/judge_distill_%A_%a.out

module purge
source activate torch-multimodal
SEEDS=(42 1 2); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python scripts/finetune_judge.py \
    --judge_model meta-llama/Llama-3.1-8B \
    --loss listnet --lambda_distill 0.5 --seed $SEED \
    --teacher_scores results/teacher_scores.json \
    --train_data esnli_human \
    --out results/judge_distill/listnet_distill_seed${SEED}/result.json
