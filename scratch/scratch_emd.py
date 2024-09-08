import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment


def emd_approx(x, y):
    print(x.shape, y.shape)
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        print(i)
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    print(emd_torch.shape)
    return emd_torch


from __future__ import print_function, division

import sys
import torch


def emd_approx_torch(x, y):
    print(x.shape, y.shape)
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

    emd_lst = []
    dist_np = dist  # .cpu().detach().numpy()
    for i in range(bs):
        print(i)
        d_i = dist_np[i]
        r_idx, c_idx = auction_lap(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    print(emd_torch.shape)
    return emd_torch


def auction_lap(X, eps=None, compute_score=True):
    """
    X: n-by-n matrix w/ integer entries
    eps: "bid size" -- smaller values means higher accuracy w/ longer runtime
    """
    eps = 1 / X.shape[0] if eps is None else eps

    # --
    # Init

    cost = torch.zeros((1, X.shape[1]))
    curr_ass = torch.zeros(X.shape[0]).long() - 1
    bids = torch.zeros(X.shape)

    if X.is_cuda:
        cost, curr_ass, bids = cost.cuda(), curr_ass.cuda(), bids.cuda()

    counter = 0
    while (curr_ass == -1).any():
        counter += 1

        # --
        # Bidding

        unassigned = (curr_ass == -1).nonzero().squeeze()

        value = X[unassigned] - cost
        top_value, top_idx = value.topk(2, dim=1)

        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]

        bid_increments = first_value - second_value + eps

        bids_ = bids[unassigned]
        bids_.zero_()
        bids_.scatter_(
            dim=1,
            index=first_idx.contiguous().view(-1, 1),
            src=bid_increments.view(-1, 1),
        )

        # --
        # Assignment

        have_bidder = (bids_ > 0).int().sum(dim=0).nonzero()

        high_bids, high_bidders = bids_[:, have_bidder].max(dim=0)
        high_bidders = unassigned[high_bidders.squeeze()]

        cost[:, have_bidder] += high_bids

        curr_ass[(curr_ass.view(-1, 1) == have_bidder.view(1, -1)).sum(dim=1)] = -1
        curr_ass[high_bidders] = have_bidder.squeeze()

    score = None
    if compute_score:
        score = int(X.gather(dim=1, index=curr_ass.view(-1, 1)).sum())

    return score, curr_ass, counter


def earth_mover_distance(y_true, y_pred):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)),
        dim=-1,
    )


a = torch.rand(size=(50, 2048, 3))  # .cuda()
b = torch.rand(size=(50, 2048, 3))  # .cuda()


print(emd_approx(a, b))
print(earth_mover_distance(a, b))
