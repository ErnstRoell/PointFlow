"""
A wrapper to give our model the same signature as the 
one in pointflow and make it accept the same type of data.

"""

from torch_geometric.data import Data, Batch
import torch

DEVICE = "cuda:0"

class ShapeNetModel:
    def __init__(self, noise=0.0, normalized=False) -> None:
        self.normalized = normalized
        self.noise = noise
    
    @torch.no_grad()
    def reconstruct(self, x, num_points=2048):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """
        x_noisy = x + self.noise * torch.randn_like(x)
        return x_noisy

class TopologicalModelVAE:
    def __init__(self, encoder, vae,normalized=False) -> None:
        self.encoder = encoder
        self.vae = vae
        self.normalized = normalized
    
    @torch.no_grad()
    def sample(self, B, N):
        """
        The way we expect the input
        B is the number of point clouds
        N is the number of points per cloud.
        _, out_pc = model.sample(B, N)
        """
        ect_samples = self.vae.model.sample(B, "cuda:0")

        # Rescale to 0,1
        ect_samples = (ect_samples + 1) / 2

        vae_pointcloud = self.encoder(ect_samples).view(B, N, 3)
        return ect_samples, vae_pointcloud

    @torch.no_grad()
    def reconstruct(self, x, num_points=2048):
        """
        Takes in a pointcloud of the form BxPxD
        and does a full reconstruction into
        a pointcloud of the form BxPxD using our model.

        We follow the PointFlow signature to make it compatible with
        their framework.
        """

        # if self.normalized: 
        #     x_means = x.mean(axis=1, keepdim=True)
        #     x_tmp = x - x_means
        #     x_norms = torch.norm(x_tmp, dim=2).max(axis=1)[0].reshape(-1, 1, 1)

        batch = Batch.from_data_list([Data(x=pts.view(-1, 3)) for pts in x])

        batch = batch.to(DEVICE)
        ect = self.encoder.layer(batch, batch.batch)
        ect = 2 * ect - 1
        reconstructed_ect, _, _, _ = self.vae(ect.unsqueeze(1))

        # Rescale to 0,1
        reconstructed_ect = (reconstructed_ect + 1) / 2

        vae_pointcloud = self.encoder(reconstructed_ect).view(-1, num_points, 3)

        # if self.normalized: 
        #     vae_pointcloud = vae_pointcloud * x_norms
        #     vae_pointcloud = vae_pointcloud + x_means


        return vae_pointcloud


class TopologicalModelEncoder:
    def __init__(self, encoder_model, normalized=False) -> None:
        super().__init__()
        self.encoder_model = encoder_model
        self.normalized = normalized

    def reconstruct(self, x, num_points=2048):
        # if self.normalized: 
        #     x_means = torch.mean(x, axis=-2)
        #     x = x - x_means.unsqueeze(1)
        #
        #     x_norms = torch.norm(x, dim=-1).max(axis=1)[0].reshape(-1, 1, 1)
        #     x = x / x_norms


        batch = Batch.from_data_list([Data(x=pts.view(-1, 3)) for pts in x])

        batch = batch.to(DEVICE)
        ect = self.encoder_model.layer(batch, batch.batch)
        encoder_pointcloud = self.encoder_model(ect).view(-1, num_points, 3)

        # if self.normalized: 
        #     encoder_pointcloud = encoder_pointcloud * x_norms
        #     encoder_pointcloud = encoder_pointcloud + x_means.unsqueeze(1)
        return encoder_pointcloud
