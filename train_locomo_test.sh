#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --disable-flash-attn \
# --reward-metric llm_judge \
# --resume-new-wandb-run
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
    --inner-epochs 4 \
    --outer-epochs 4 \
    --batch-size 4 \
    --encode-batch-size 64 \
    --session-mode full-session \
    --ppo-epochs 2 \
    --action-top-k 3 \
    --new-action-bias-steps 25 \
    --stage-reward-fraction 0.25 \
    --designer-reflection-cycles 3 \
    --mem-top-k 20 \
    --mem-top-k-eval 20 \
    --designer-max-changes 3 \
    --designer-failure-window-epochs 100 \
    --designer-failure-pool-size 2000 \
    --reward-metric f1 \
    --designer-new-skill-hint \
    --device cuda \
    --enable-designer \
    --wandb-run-name locomo-train_test0329 \
    --save-dir ./checkpoints/locomo-train_test0329 \
    --out-file ./results/locomo-train_test0329.json \
    "$@"
