import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import librosa
import pytorch_lightning as pl
import wandb
import torchaudio
import plotly.graph_objs as go
from torch import Tensor
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger  # , LoggerCollection
from audio_diffusion_pytorch import DiffusionModel
from audio_encoders_pytorch import Encoder1d
from typing import List, Optional, Callable
from einops import rearrange
from main.utils import int16_to_float32, float32_to_int16


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        model: DiffusionModel,
        onsets_encoder: Encoder1d,
        embedder: torch.nn.Module,
        embedder_checkpoint: str
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.model: DiffusionModel = model

        # Onsets encoder
        self.onsets_encoder: Encoder1d = onsets_encoder

        # Clap embedder
        self.clap = embedder
        self.clap.load_ckpt(embedder_checkpoint)
        for param in self.clap.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) +
            list(self.onsets_encoder.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def clap_encode_audio(self, x: Tensor) -> Tensor:
        x = int16_to_float32(float32_to_int16(x[:, 0, :])).float()
        return self.clap.get_audio_embedding_from_data(x=x, use_tensor=True).unsqueeze(1)

    @torch.no_grad()
    def clap_encode_text(self, text: List[str]) -> Tensor:
        return self.clap.get_text_embedding(text, use_tensor=True).unsqueeze(1)

    def step(self, batch):
        x, y, z, _ = batch
        # z = pad_sequence([random.choice(z) for z in zs],
        #                 batch_first=True,
        #                 padding_value=0.)
        z_latent = self.clap_encode_audio(z)
        _, y_latent = self.onsets_encoder(y, with_info=True)
        return self.model(x, channels=y_latent['xs'][2:-1], embedding=z_latent)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("valid_loss", loss)
        return loss


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    # if isinstance(trainer.logger, LoggerCollection):
    #     for logger in trainer.logger:
    #         if isinstance(logger, WandbLogger):
    #             return logger

    print("WandbLogger not found.")
    return None


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )


def log_wandb_audio_spectrogram(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = samples.detach().cpu()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = librosa.power_to_db(spectrogram)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_text=caption,
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        length: int,
        sampling_steps: List[int],
        embedding_scale: float
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.length = length
        self.sampling_steps = sampling_steps
        self.log_next = False
        self.embedding_scale = embedding_scale

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment
        diffusion_model: DiffusionModel = pl_module.model
        onsets_encoder: Encoder1d = pl_module.onsets_encoder

        # Get start diffusion noise
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )
        _, y, z, _ = batch
        _, y_latent = onsets_encoder(y[:self.num_items, :, :], with_info=True)
        z_latent = pl_module.clap_encode_audio(z[:self.num_items, :, :])

        for steps in self.sampling_steps:
            samples = diffusion_model.sample(
                x_noisy=noise,
                num_steps=steps,
                channels=y_latent['xs'][2:-1],
                embedding=z_latent,
                embedding_scale=self.embedding_scale,
            )

            log_wandb_audio_batch(
                logger=wandb_logger,
                id="sample",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )

            log_wandb_audio_batch(
                logger=wandb_logger,
                id="onsets",
                samples=y[:self.num_items, :, :],
                sampling_rate=self.sampling_rate,
                caption=f"Onsets",
            )

            log_wandb_audio_batch(
                logger=wandb_logger,
                id="cond",
                samples=z,
                sampling_rate=self.sampling_rate,
                caption=f"Conditioning",
            )

            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id="sample",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )

            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id="onsets",
                samples=y[:self.num_items, :, :],
                sampling_rate=self.sampling_rate,
                caption=f"Onsets",
            )

            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id="cond",
                samples=z,
                sampling_rate=self.sampling_rate,
                caption=f"Conditioning",
            )

        if is_train:
            pl_module.train()
