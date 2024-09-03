import torch


def normalize(pts):
    assert pts.shape[1:] == (2048, 3)
    pts_means = pts.mean(axis=-2, keepdim=True)
    pts = pts - pts_means
    pts_norms = torch.norm(pts, dim=-1, keepdim=True).max(dim=-2, keepdim=True)[0]
    pts = pts / pts_norms
    return pts


if __name__ == "__main__":
    normalize(torch.rand(10, 2048, 3))


# ########################
# ## Insert
# ########################

# out_pc = normalize(out_pc)
# te_pc = normalize(te_pc)

# ########################
# ## End insert
# ########################
