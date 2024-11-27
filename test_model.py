import json
from pprint import pprint
from datasets import get_test_loader, synsetid_to_cate
from args import get_args
import torch

import torch.nn as nn

# from metrics.evaluation_metrics import compute_all_metrics
from models.networks import PointFlow
from models.vae import BaseModel as VAE
from models.encoder import BaseModel as Encoder
from models.encoder_scaled import BaseModel as EncoderScaled
from models.encoder_extended import BaseModel as EncoderExtended
from model_wrapper import (
    TopologicalModelVAE,
    TopologicalModelEncoder,
    ShapeNetModel,
    TopologicalModelEncoderScaled,
)
from models.networks import PointFlow
from load_model_scaled import load_encoder, load_vae


from normalization import normalize


args = get_args()

# print("Loading VAE Scaled")
# path = f"./configs/config_encoder_shapenet_{args.cates[0]}_scaled.yaml"
# encoder = load_encoder(path)
path = f"./configs/config_vae_shapenet_{args.cates[0]}.yaml"
encoder = load_vae(path)

# model = TopologicalModelVAE(encoder_model, vae)
# model.vae.eval()
