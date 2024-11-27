#! /bin/bash 

# Categories
declare -a models=("PointFlow" "VAE" "VAEScaled")
# declare -a models=("SetVAE" "SoftFlow" "ShapeGF")
declare -a arr=("airplane" "car" "chair")

fast_run="--fast_run"

for model in "${models[@]}"
do 

  # Loop over the categories 
  for cate in "${arr[@]}"
  do
      echo "$model $cate"
      python3 compute_generative_metrics.py \
          --cate $cate \
          --model $model \
          --normalize \
          $fast_run 

      python3 compute_generative_metrics.py \
          --cate $cate \
          --model $model \
          $fast_run 
  done
done



