#! /bin/bash
python3 test.py --cates airplane --resume_checkpoint pretrained_models/ae/airplane/checkpoint.pt --dims 512-512-512 --use_deterministic_encoder --evaluate_recon

python3 test.py --cates chair --resume_checkpoint pretrained_models/ae/chair/checkpoint.pt --dims 512-512-512 --use_deterministic_encoder --evaluate_recon

python3 test.py --cates car --resume_checkpoint pretrained_models/ae/car/checkpoint.pt --dims 512-512-512 --use_deterministic_encoder --evaluate_recon
