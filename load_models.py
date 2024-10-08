from models.vae import VanillaVAE
from models.vae import BaseModel as BaseVAE

from models.encoder import BaseModel as EctEncoder

from metrics.metrics import get_mse_metrics
from metrics.accuracies import compute_mse_accuracies
from metrics.loss import compute_mse_loss_fn

from layers.ect import EctLayer, EctConfig
from layers.directions import generate_directions

from lightning.pytorch.utilities.model_summary import ModelSummary


def load_encoder(encoder_config, device="cuda:0"):
    layer = EctLayer(
        EctConfig(
            num_thetas=encoder_config.layer.ect_size,
            bump_steps=encoder_config.layer.ect_size,
            # r=1,
            normalized=True,
            device=device,
        ),
        v=generate_directions(
            encoder_config.layer.ect_size, encoder_config.layer.dim, device
        ),
    )
    # Load the encoder

    ect_encoder_litmodel = EctEncoder.load_from_checkpoint(
        checkpoint_path=f"./trained_models/{encoder_config.model.save_name}",
        layer=layer,
        ect_size=encoder_config.layer.ect_size,
        hidden_size=encoder_config.model.hidden_size,
        num_pts=encoder_config.model.num_pts,
        num_dims=encoder_config.model.num_dims,
        learning_rate=encoder_config.model.learning_rate,
    ).to(device)

    print(ModelSummary(ect_encoder_litmodel))

    return ect_encoder_litmodel


def load_vae(vae_config, device="cuda:0"):
    layer = EctLayer(
        EctConfig(
            num_thetas=vae_config.layer.ect_size,
            bump_steps=vae_config.layer.ect_size,
            normalized=True,
            # r=1,
            device=device,
        ),
        v=generate_directions(vae_config.layer.ect_size, vae_config.layer.dim, device),
    )

    vae_model = VanillaVAE(
        in_channels=vae_config.model.in_channels,
        latent_dim=vae_config.model.latent_dim,
        img_size=vae_config.layer.ect_size,
    )
    metrics = get_mse_metrics()

    vae_litmodel = BaseVAE.load_from_checkpoint(
        f"./trained_models/{vae_config.model.save_name}",
        model=vae_model,
        training_accuracy=metrics[0],
        test_accuracy=metrics[1],
        validation_accuracy=metrics[2],
        accuracies_fn=compute_mse_accuracies,
        loss_fn=compute_mse_loss_fn,
        learning_rate=0.01,
        layer=layer,
    ).to(device)
    return vae_litmodel


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("./configs/config_encoder_modelnet.yaml")

    encoder = load_encoder(config)
