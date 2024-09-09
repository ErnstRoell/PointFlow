#! /bin/bash

# Normalized evaluation 
python3 test.py \
    --cates $cate \
    --resume_checkpoint pretrained_models/ae/$cate/checkpoint.pt \
    --dims 512-512-512 \
    --use_deterministic_encoder \
    --evaluate_recon \
    --normalize \
    --num_reruns $num_reruns \
    --model $model \
      $fast_run 
