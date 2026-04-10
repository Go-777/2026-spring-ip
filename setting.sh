#!/bin/bash
#SBATCH -A mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --mail-user=zzhaodg@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# 激活环境
source ~/.bashrc
conda activate memskill
cd ~/MemSkill
python main.py \
    --dataset locomo \
    --data-file "data/locomo10.json" \
    --model "qwen/qwen3-8b" \
    --designer-model "qwen/qwen3-8b" \
    --api \
    --api-base "https://openrouter.ai/api/v1" \
    --api-key "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" \
    --retriever contriever \
    --designer-freq 1 \
    --inner-epochs 20 \
    --outer-epochs 8 \
    --batch-size 4 \
    --encode-batch-size 64 \
    --session-mode full-session \
    --ppo-epochs 2 \
    --action-top-k 3 \
    --new-action-bias-steps 25 \
    --stage-reward-fraction 0.25 \
    --designer-reflection-cycles 3 \
    --mem-top-k 10 \
    --mem-top-k-eval 10 \
    --designer-max-changes 3 \
    --designer-failure-window-epochs 100 \
    --designer-failure-pool-size 2000 \
    --reward-metric f1 \
    --designer-new-skill-hint \
    --device cuda \
    --enable-designer \
    --wandb-run-name locomo-train \
    --save-dir ./checkpoints/locomo_with_designer \
    --out-file ./results/locomo_with_designer.json \
    --load-checkpoint ./checkpoints/locomo_with_designer/locomo-train_epoch_2.pt
    "$@"