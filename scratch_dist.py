''' '''
import torch
from kaolin.metrics.pointcloud import chamfer_distance
from metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
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
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]





a = torch.rand(size=(3, 50, 3)).cuda()
b = torch.rand(size=(3, 50, 3)).cuda()

dl, dr = distChamfer(a, b)
ch = dl.mean(axis=1) + dr.mean(axis=1)

ch2 = chamfer_distance(a.cuda(), b.cuda())

dl1,dr1,_,_ = cham3D(a,b)
ch3 = dl1.mean(axis=1) + dr1.mean(axis=1)



assert torch.allclose(ch,ch2)
assert torch.allclose(ch,ch3)

print(ch)
print(ch2)
print(ch3)
