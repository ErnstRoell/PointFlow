#! /bin/bash

# Parameters
num_reruns=1
cate="chair"
model="EncoderSparse"
rm -rf results/$model 
mkdir results/$model
# python3 test.py \
#     --cates $cate \
#     --resume_checkpoint pretrained_models/ae/$cate/checkpoint.pt \
#     --dims 512-512-512 \
#     --use_deterministic_encoder \
#     --evaluate_recon \
#     --normalize \
#     --num_reruns $num_reruns \
#     --model $model 
    
python3 test.py \
    --cates $cate \
    --resume_checkpoint pretrained_models/ae/$cate/checkpoint.pt \
    --dims 512-512-512 \
    --use_deterministic_encoder \
    --evaluate_recon \
    --num_reruns $num_reruns \
    --model $model 
