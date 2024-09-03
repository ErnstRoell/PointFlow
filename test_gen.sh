#! /bin/bash 

echo "Test generation pointflow"
python3 test.py \
    --cates airplane \
    --resume_checkpoint pretrained_models/gen/airplane/checkpoint.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_latent_flow

python3 test.py \
    --cates car \
    --resume_checkpoint pretrained_models/gen/car/checkpoint.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_latent_flow

python3 test.py \
    --cates chair \
    --resume_checkpoint pretrained_models/gen/chair/checkpoint.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_latent_flow
