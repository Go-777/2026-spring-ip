#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --disable-flash-attn \
# --reward-metric llm_judge \
# --resume-new-wandb-run
# --mem-top-k 20 \
# --mem-top-k-eval 20 \
# --inner-epochs 100 \
# --outer-epochs 10 \
# train_locomo.sh --load-checkpoint ./checkpoints/locomo_with_designer/locomo-train_epoch_1.pt
python main.py \
    --dataset locomo \
    --data-file "data/locomo10.json" \
    --model "qwen/qwen3-8b" \
    --designer-model "qwen/qwen3-8b" \
    --api \
    --api-base "https://openrouter.ai/api/v1" \
    --api-key "" \
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
