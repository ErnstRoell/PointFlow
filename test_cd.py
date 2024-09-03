import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

from metrics.PyTorchEMD.emd import earth_mover_distance as EMD
from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.ChamferDistancePytorch.fscore import fscore
from tqdm import tqdm

from kaolin.metrics.pointcloud import chamfer_distance

cham3D = chamfer_3DDist()


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


if __name__ == "__main__":
    a = torch.rand(10, 2048, 3,requires_grad=True).cuda()
    b = torch.rand(10, 2048, 3).cuda()
    print(EMD(a, b,transpose=False))
    # dl,dr = distChamfer(a,b)
    # cd = (dl + dr).mean()
    # print(cd.shape)
    # cd1 = chamfer_distance(a,b)
    # print(cd1.shape)
    # assert torch.allclose(cd,cd1)
