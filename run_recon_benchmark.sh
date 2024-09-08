#! /bin/bash 



# Categories
declare -a arr=("airplane" "car" "chair")

# Parameters
num_reruns=10
model="PointFlow"
# fast_run="--fast_run"
fast_run=""

mkdir -p results
rm -rf results/$model
mkdir results/$model

# Loop over the categories 
for cate in "${arr[@]}"
do
    echo "$cate"

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

    # Non-normalized evaluation 
    python3 test.py \
        --cates $cate \
        --resume_checkpoint pretrained_models/ae/$cate/checkpoint.pt \
        --dims 512-512-512 \
        --use_deterministic_encoder \
        --evaluate_recon \
        --num_reruns $num_reruns \
        --model $model \
        $fast_run
done
