#! /bin/bash



# Categories
declare -a arr=("airplane" "car" "chair")
declare -a models=("ShapeNet-0" "ShapeNet-005" "ShapeNet-01")

# Parameters
num_reruns=10
# fast_run="--fast_run"
fast_run=""


for model in "${models[@]}"
do
  rm -rf results/$model 
  mkdir results/$model

  for cate in "${arr[@]}"
  do
    # Normalized evaluation 
    python3 test.py \
        --cates $cate \
        --resume_checkpoint pretrained_models/ae/$cate/checkpoint.pt \
        --dims 512-512-512 \
        --use_deterministic_encoder \
        --evaluate_recon \
        --normalize \
        --num_reruns $num_reruns \
        --model $model 
    
    # Normalized evaluation 
    python3 test.py \
        --cates $cate \
        --resume_checkpoint pretrained_models/ae/$cate/checkpoint.pt \
        --dims 512-512-512 \
        --use_deterministic_encoder \
        --evaluate_recon \
        --num_reruns $num_reruns \
        --model $model 
    done
done
