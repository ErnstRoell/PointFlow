from datasets import get_datasets, synsetid_to_cate
from args import get_parser
from pprint import pprint
from collections import defaultdict

import os
import torch
import numpy as np
import torch.nn as nn

# import pyvista as pv


from omegaconf import OmegaConf
from model_wrapper import TopologicalModelEncoder
from load_models import load_encoder, load_vae


def get_test_loader(args):
    _, te_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        te_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def get_train_loader(args):
    tr_dataset, te_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        raise "hello"
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        te_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader


p = get_parser()
args, unknown = p.parse_known_args()
args.cates = ["airplane"]

import json

with open("args.json", "w") as f:
    json.dump(vars(args), f)

# cates = args.cates

# all_results = {}
# cate_to_len = {}

# for cate in cates:
#     args.cates = [cate]
#     loader = get_train_loader(args)

#     all_sample = []
#     all_ref = []
#     for data in loader:
#         idx_b, tr_pc, te_pc = data["idx"], data["train_points"], data["test_points"]
#         te_pc = te_pc.cuda()  # if args.gpu is None else te_pc.cuda(args.gpu)
#         tr_pc = tr_pc.cuda()  # if args.gpu is None else tr_pc.cuda(args.gpu)
#         B, N = te_pc.size(0), te_pc.size(1)
#         print(tr_pc.norm(dim=-1).max().item())
#         # print(tr_pc.mean(dim=-2).norm(dim=-1).max())


# import glob

# folder = "./data/ShapeNetCore.v2.PC15k/02691156/train"
# files = glob.glob("./data/ShapeNetCore.v2.PC15k/02691156/train/*.npy")

# plane = torch.vstack([torch.tensor(np.load(f)) for f in files])
# print(plane.shape)
# print(plane.norm(dim=-1).max())
