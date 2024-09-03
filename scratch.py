import numpy as np
from datasets import get_datasets, synsetid_to_cate
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn


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


reference_points = torch.tensor(np.load("./airplane_out_ref.npy"))
smp_points = torch.tensor(np.load("./airplane_out_smp.npy"))


print(smp_points.mean(axis=1, keepdim=True))


# ref_norms = torch.norm(reference_points, dim=2).max(axis=1)[0].view(-1, 1, 1)
# reference_points = reference_points / ref_norms
# smp_points = smp_points / ref_norms


# ref_norms = torch.norm(reference_points, dim=2).max(axis=1)[0]

# smp_norms = torch.norm(smp_points, dim=2).max(axis=1)[0]

# print(np.vstack([smp_norms, ref_norms]).T[:10])


# print(80 * "=")
# print(ref_norms.max())
# print(ref_norms.min())
# print(ref_norms.mean())
# print(80 * "=")
# print(smp_norms.max())
# print(smp_norms.min())
# print(smp_norms.mean())


# args = get_args()

# loader = get_test_loader(args)

# for data in loader:
#     idx_b, tr_pc, te_pc = data["idx"], data["train_points"], data["test_points"]
#     print(tr_pc.shape)
#     n = np.linalg.norm(tr_pc, axis=2).max(axis=1)
#     print("max", n.max())
#     print(n.min())
#     print(n.mean())
# 9
