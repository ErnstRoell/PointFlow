from layers.ect import EctConfig, EctLayer
import matplotlib.pyplot as plt
from torch_geometric.data import Batch, Data


import torch
from torch import nn
import torch.nn.functional as F


def generate_uniform_directions(num_thetas: int = 64, d: int = 3):
    """
    Generate randomly sampled directions from a sphere in d dimensions.

    First a standard gaussian centered at 0 with standard deviation 1 is sampled
    and then projected onto the unit sphere. This yields a uniformly sampled set
    of points on the unit spere. Please note that the generated shapes with have
    shape [d, num_thetas].

    Parameters
    ----------
    num_thetas: int
        The number of directions to generate.
    d: int
        The dimension of the unit sphere. Default is 3 (hence R^3)
    """
    v = torch.randn(size=(d, num_thetas))
    v /= v.pow(2).sum(axis=0).sqrt().unsqueeze(1).T
    return v


def compute_topo_kld_loss_fn(decoded, mu=None, log_var=None, ect=None):
    # kld_loss = torch.mean(
    #     -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    # )

    # Clip
    decoded = torch.clamp(decoded, min=0)
    decoded /= decoded.sum(axis=1, keepdim=True)
    ect /= ect.sum(axis=1, keepdim=True)

    eps = 10e-5
    decoded += eps
    ect += eps

    ect_kld_loss = (
        F.kl_div(decoded.log(), ect, None, None, reduction="none")
        .sum(dim=-1)
        .sum(dim=-1)
        / 2048
    )

    return ect_kld_loss  # + kld_loss


batch_pred = Batch.from_data_list(
    [Data(x=0.5 * torch.rand(size=(2048, 3))) for _ in range(8)]
).cuda()


batch_target = Batch.from_data_list(
    [Data(x=0.6 * torch.rand(size=(2048, 3))) for _ in range(8)]
).cuda()
ECT_SIZE = 64

v = generate_uniform_directions(num_thetas=ECT_SIZE).cuda()


loss_layer = EctLayer(
    EctConfig(
        bump_steps=ECT_SIZE,
        num_thetas=ECT_SIZE,
        device="cuda:0",
        ect_type="points_derivative",
        normalized=True,
    ),
    v=v,
)

ect_pred = loss_layer(batch_pred, batch_pred.batch).cpu().detach().squeeze()
ect_target = loss_layer(batch_target, batch_target.batch).cpu().detach().squeeze()


eps = 10e-5
ect_pred /= ect_pred.sum(axis=1, keepdim=True)
ect_target /= ect_target.sum(axis=1, keepdim=True)
ect_pred += eps
ect_target += eps

d = (
    F.kl_div(ect_pred.log(), ect_target, None, None, reduction="none")
    .sum(dim=-1)
    .sum(dim=-1)
    / 2048
)
print(d)

loss = compute_topo_kld_loss_fn(
    torch.randn_like(ect_target, requires_grad=True), ect=ect_target
).mean()
print(loss)
loss.backward()

# decoded = torch.randn_like(ect_pred)
# ect = ect_target
# # print(decoded[0, :, 0])
# # Clip
# decoded = torch.clamp(decoded, min=0)
# decoded /= decoded.sum(axis=1, keepdim=True)
# ect /= ect.sum(axis=1, keepdim=True)

# eps = 10e-5
# decoded += eps
# ect += eps

# # print(decoded[0, :, 0])
# # print(decoded[0, :, 1].sum())


# ect_kld_loss = (
#     F.kl_div(decoded.log(), ect, None, None, reduction="none").sum(dim=-1).sum(dim=-1)
#     / 2048
# )

# print(ect_kld_loss)


# print(ect[0].min())
# print(ect[0].max())

# print(ect[1].min())
# print(ect[1].max())

# print(d)
# def gaussian(x, mean=0, sigma=1):
#     return torch.exp(-1 * (x - mean) ** 2 / (2 * sigma**2))


# P = (
#     gaussian(torch.linspace(-10, 10, 200), mean=0.3)
#     + gaussian(torch.linspace(-10, 10, 200), mean=-2)
#     + gaussian(torch.linspace(-10, 10, 200), mean=5)
# )
# P = P / P.sum()
# Q = gaussian(torch.linspace(-10, 10, 200), mean=1.0)
# Q = Q / Q.sum()


# # plt.plot(Q)
# # plt.plot(P)
# # plt.show()

# print((P - Q).pow(2).sum())


# print(d)


# # kl_loss = nn.KLDivLoss(reduction="batchmean")
# # input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
# # target = F.softmax(torch.rand(3, 5), dim=1)
# # output = kl_loss(input, target)


# # print(input)
# # print(target)
# # print(output)
