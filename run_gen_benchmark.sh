#! /bin/bash 

# Categories
declare -a models=("PointFlow" "VAE")
declare -a arr=("airplane" "car" "chair")

# Parameters
num_reruns=5

fast_run="--fast_run"

for model in "${models[@]}"
do 

  rm -rf results_gen/$model
  mkdir -p results_gen/$model

  # Loop over the categories 
  for cate in "${arr[@]}"
  do
      echo "$model $cate"
      python3 test.py \
          --cates $cate \
          --resume_checkpoint pretrained_models/gen/$cate/checkpoint.pt \
          --dims 512-512-512 \
          --latent_dims 256-256 \
          --use_latent_flow \
          --num_reruns $num_reruns \
          --model $model \
          --fast_run 
  done
done
