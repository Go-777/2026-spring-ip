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

export CUDA_VISIBLE_DEVICES=0

# --disable-flash-attn \
# NOTE: chunk-size 256 for Locomo + Qwen
source ~/.bashrc
conda activate memskill
cd ~/MemSkill
python main.py \
    --memory-cache-suffix "locomo_eval" \
    --eval-only \
    --inference-workers 4 \
    --inference-session-workers 1 \
    --action-top-k 7 \
    --mem-top-k-eval 20 \
    --session-mode fixed-length \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --load-checkpoint './checkpoints/locomo_with_designer/locomo-train_epoch_final.pt' \
    --dataset locomo \
    --data-file "data/locomo10.json" \
    --model "qwen/qwen3-8b" \
    --api \
    --api-base "https://openrouter.ai/api/v1" \
    --api-key "" \
    --retriever contriever \
    --designer-freq 1 \
    --inner-epochs 20 \
    --outer-epochs 8 \
    --batch-size 4 \
    --encode-batch-size 64 \
    --ppo-epochs 2 \
    --new-action-bias-steps 25 \
    --stage-reward-fraction 0.25 \
    --designer-reflection-cycles 3 \
    --mem-top-k 20 \
    --designer-max-changes 2 \
    --designer-failure-window-epochs 100 \
    --designer-failure-pool-size 2000 \
    --reward-metric llm_judge \
    --designer-new-skill-hint \
    --device cuda \
    --enable-designer \
    --skip-load-snapshot-manager \
    --wandb-run-name eval \
    --save-dir ./checkpoints/locomo_with_designer_eval \
    --out-file ./results/locomo_with_designer_eval.json
