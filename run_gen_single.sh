#! /bin/bash 

# Categories
declare -a models=("VAE" "VAEScaled" "PointFlow")
declare -a arr=("airplane" "car" "chair")

# Parameters
num_reruns=5


for model in "${models[@]}"
do 

  rm -rf results_gen/$model
  mkdir -p results_gen/$model

  # for i in $(seq 0 $(($num_reruns - 1)))
  # do 
  #   mkdir -p results_gen/$model/$i 
  # done 

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
          --normalize

      python3 test.py \
          --cates $cate \
          --resume_checkpoint pretrained_models/gen/$cate/checkpoint.pt \
          --dims 512-512-512 \
          --latent_dims 256-256 \
          --use_latent_flow \
          --num_reruns $num_reruns \
          --model $model 
  done
done

