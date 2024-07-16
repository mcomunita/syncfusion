from pathlib import Path
from typing import Union, Optional

import torchaudio
import webdataset as wds
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from main.dataset_diffusion import collate_fn

@torch.no_grad()
def generate_dataset(
        experiment_path: Union[str, Path],
        model: LightningModule,
        dataset: wds.WebDataset,
        device: str = "cuda",
        model_path: str = None,
        batch_size: int = 16,
        num_workers: int = 4,
        sample_rate: int = 48000,
        num_steps: int = 150,
        length: int = 2**18,
        embedding_scale: float = 7.5,
        cut_prefix: bool = False,
        cond_text: bool = False,
        one_chunk_per_track: bool = False,
        cut_length: Optional[int] = None,
        downsample_rate: Optional[int] = None,
        save_cond: bool = False
):
    if not cut_length:
        cut_length = length
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(exist_ok=True, parents=True)

    # Get samples
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=collate_fn)

    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # Main generation loop
    chunk_id = 0
    for batch_idx, batch in enumerate(loader):
        x, y, z, text, filenames = batch

        if not one_chunk_per_track:
            last_chunk_batch_id = chunk_id + batch[0].shape[0] - 1
            last_chunk_path = experiment_path / f"{last_chunk_batch_id}.wav"

            if last_chunk_path.exists():
                print(f"Skipping path: {last_chunk_path}")
                chunk_id = chunk_id + batch[0].shape[0]
                continue
            print(f"{chunk_id = }")
        else:
            last_filename = filenames[-1].split("/")[-1]
            last_filename_path = experiment_path / f"{last_filename}.wav"
            if last_filename_path.exists():
                print(f"Skipping path: {last_filename_path}")
            print(f"{filenames[0] = }")

        # Get start diffusion noise
        noise = torch.randn((x.shape[0], 1, length), device=model.device)
        y, z = y.to(device), z.to(device)
        _, y_latent = model.onsets_encoder(y, with_info=True)
        if cond_text:
            z_latent = model.clap_encode_text(text)
        else:
            z_latent = model.clap_encode_audio(z)

        gen = model.model.sample(
                x_noisy=noise.to(device),
                num_steps=num_steps,
                channels=y_latent['xs'][2:-1],
                embedding=z_latent.to(device),
                embedding_scale=embedding_scale
            )

        # Save generated audio
        for i in range(gen.shape[0]):
            if cut_prefix:
                first_onset = torch.nonzero(y[i][0]).squeeze(-1)[0]
                gen[i, :, :first_onset] = 0.
            if downsample_rate:
                output = torchaudio.functional.resample(gen[i, :, :cut_length].cpu(),
                                                        orig_freq=sample_rate,
                                                        new_freq=downsample_rate)
                if save_cond and not cond_text:
                    output_cond = torchaudio.functional.resample(z[i].cpu(),
                                                                 orig_freq=sample_rate,
                                                                 new_freq=downsample_rate)
                output_sr = downsample_rate
            else:
                output = gen[i, :, :cut_length].cpu()
                if save_cond and not cond_text:
                    output_cond = z[i].cpu()
                output_sr = sample_rate
            if not one_chunk_per_track:
                if save_cond:
                    if cond_text:
                        torchaudio.save(experiment_path / f"{chunk_id}_{text[i]}.wav", output, sample_rate=output_sr)
                    else:
                        torchaudio.save(experiment_path / f"{chunk_id}.wav", output, sample_rate=output_sr)
                        torchaudio.save(experiment_path / f"{chunk_id}_cond.wav", output_cond, sample_rate=output_sr)
                else:
                    torchaudio.save(experiment_path / f"{chunk_id}.wav", output, sample_rate=output_sr)
                chunk_id += 1
            else:
                if save_cond:
                    if cond_text:
                        torchaudio.save(experiment_path / f"{filenames[i].split('/')[-1]}_{text[i]}.wav", output, sample_rate=output_sr)
                    else:
                        torchaudio.save(experiment_path / f"{filenames[i].split('/')[-1]}.wav", output, sample_rate=output_sr)
                        torchaudio.save(experiment_path / f"{filenames[i].split('/')[-1]}_cond.wav", output_cond, sample_rate=output_sr)
                else:
                    torchaudio.save(experiment_path / f"{filenames[i].split('/')[-1]}.wav", output, sample_rate=output_sr)

