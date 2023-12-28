from functools import partial
import random
from math import ceil
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset.autodecode import torch_audio
from torchaudio.functional import resample


def _fn_resample(sample, sample_rate):
    return {k: (resample(v[0], orig_freq=v[1], new_freq=sample_rate), sample_rate) if k.endswith("wav") else v for k, v in sample.items()}


def _decode_csv(sample):
    def decode(k):
        csv_data = sample[k].decode('utf-8').split('\n')[:-1]
        splitted = [item.split(',') for item in csv_data]
        result_dict = {float(item[0]): item[1] if len(item) > 1 else None for item in splitted}
        return result_dict
    return {k: decode(k) if k.endswith("csv") else v for k, v in sample.items()}


def _to_tuple(sample):
    wav = sample["resampled.wav"]
    onset_metadata = sample["times.csv"]
    pred_onset_metadata = sample['times.pred.csv'] if 'times.pred.csv' in sample else None
    filename = sample['__key__']
    # remaining_data = [v for k, v in sample.items() if k.endswith("wav") and k != "wav" and k != "onset.wav"]
    return wav, onset_metadata, pred_onset_metadata, filename


def _get_cond_chunk(waveform: torch.Tensor, onset_indices) -> torch.Tensor:
    n_onsets = len(onset_indices)
    onset_i = random.randint(0, n_onsets - 1)
    start_index = onset_indices[onset_i]
    if onset_i == n_onsets - 1:
        end_index = waveform.shape[1]
    else:
        end_index = onset_indices[onset_i + 1]
    return waveform[:, start_index:end_index]

def _get_slices(src, chunk_size, onset_check_length, shift_augment=False, cut_prefix=True,
                one_chunk_per_track=False):
    for sample in src:
        done_chunk = False
        # get length of first element in step
        (wav, sr), onset_metadata, pred_onset_metadata, filename = sample
        channels, length = wav.shape

        if pred_onset_metadata is None:
            pred_onset_metadata = onset_metadata

        onset_idx = [int(k * sr) for k in onset_metadata.keys()]
        assert onset_idx
        onset = torch.zeros_like(wav)
        onset[:, onset_idx] = 1.0

        pred_onset_idx = [int(k * sr) for k in pred_onset_metadata.keys()]
        assert pred_onset_idx
        pred_onset = torch.zeros_like(wav)
        pred_onset[:, pred_onset_idx] = 1.0

        # Continue if length is smaller than chunk_size
        assert length >= chunk_size

        if shift_augment:
            max_shift = length - (length // chunk_size) * chunk_size
            shift = torch.randint(0, max_shift + 1, (1,)).item()
        else:
            shift = 0

        for i in range(length // chunk_size):
            if done_chunk and one_chunk_per_track:
                break
            start_idx = min(length - chunk_size, i * chunk_size + shift)
            end_idx = start_idx + chunk_size
            wav_chunk = wav[:, start_idx: end_idx]
            onset_chunk = onset[:, start_idx: end_idx]
            pred_onset_chunk = pred_onset[:, start_idx: end_idx]

            if torch.all(onset_chunk[:, :onset_check_length] == 0.):
                if one_chunk_per_track:
                    print(f"Skipping path ${filename} for zero onsets")
                    break
                else:
                    continue
            # assert not torch.all(onset_chunk == 0.)
            # assert not torch.all(pred_onset_chunk == 0.)

            onset_indices = torch.nonzero(onset_chunk[0]).squeeze(-1)

            # cut all to the left of first onset
            if cut_prefix:
                wav_chunk[:, :onset_indices[0]] = 0.
            cond_chunk = _get_cond_chunk(wav_chunk, onset_indices)
            done_chunk = True
            yield wav_chunk, pred_onset_chunk, cond_chunk, filename


def create_sfx_dataset(
        path: str,
        sample_rate: int,
        chunk_size: Optional[int] = None,
        shardshuffle: bool = False,
        shift_augment: bool = False,
        cut_prefix: bool = True,
        one_chunk_per_track: bool = True,
        onset_check_length: Optional[int] = None
        ):

    get_slices = partial(_get_slices, chunk_size=chunk_size, shift_augment=shift_augment, cut_prefix=cut_prefix,
                         onset_check_length=onset_check_length if onset_check_length else chunk_size,
                         one_chunk_per_track=one_chunk_per_track)
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)

    # create datapipeline
    dataset = (wds.WebDataset(path, shardshuffle=shardshuffle).decode(torch_audio).map(_decode_csv)
               .map(fn_resample).map(_to_tuple))
    dataset = dataset.compose(get_slices) if chunk_size is not None else dataset
    return dataset


def collate_fn(data):
    waveforms, onset_tensors, cond_chunks, filenames = zip(*data)
    # Stack waveforms and onset_tensors directly
    waveforms_batch = torch.stack(waveforms, dim=0)
    onset_tensors_batch = torch.stack(onset_tensors, dim=0)
    # Pad cond_chunks to the same length and then stack
    max_length = max(chunk.size(1) for chunk in cond_chunks)
    cond_chunks_padded = [torch.nn.functional.pad(chunk, (0, max_length - chunk.size(1))) for chunk in cond_chunks]
    cond_chunks_batch = torch.stack(cond_chunks_padded, dim=0)
    return waveforms_batch, onset_tensors_batch, cond_chunks_batch, filenames


def create_fad_gt(
        experiment_path: Union[str, Path],
        dataset: wds.WebDataset,
        batch_size: int = 16,
        num_workers: int = 4,
        sample_rate: int = 48000,
        one_chunk_per_track: bool = False,
        downsample_rate: Optional[int] = None
):
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(exist_ok=True, parents=True)

    # Get samples
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    # Main generation loop
    chunk_id = 0

    for batch_idx, batch in enumerate(loader):
        waveforms, _, _, filenames = batch
        if not one_chunk_per_track:
            last_chunk_batch_id = chunk_id + batch[0].shape[0] - 1
            last_chunk_path = experiment_path / f"{last_chunk_batch_id}.wav"

            if last_chunk_path.exists():
                print(f"Skipping path: {last_chunk_path}")
                chunk_id = chunk_id + batch[0].shape[0]
                continue
            print(f"{chunk_id=}")
        else:
            last_filename = filenames[-1].split("/")[-1]
            last_filename_path = experiment_path / f"{last_filename}.wav"
            if last_filename_path.exists():
                print(f"Skipping path: {last_filename_path}")
            print(f"{filenames[0] = }")

        # print(f"batch {batch_idx + 1} out of {ceil(len(dataset) / batch_size)}")

        # Save gt mixtures
        for i in range(waveforms.shape[0]):
            if downsample_rate:
                output = torchaudio.functional.resample(waveforms[i], orig_freq=sample_rate,
                                                        new_freq=downsample_rate)
                output_sr = downsample_rate
            else:
                output = waveforms[i]
                output_sr = sample_rate
            if not one_chunk_per_track:
                torchaudio.save(experiment_path / f"{chunk_id}.wav", output, sample_rate=output_sr)
                chunk_id += 1
            else:
                torchaudio.save(experiment_path / f"{filenames[i].split('/')[-1]}.wav", output, sample_rate=output_sr)


if __name__ == '__main__':
    dataset_train = create_sfx_dataset("data/DIFF-SFX-webdataset/greatest_hits/test_onset_preds.tar",
                                       sample_rate=48000, chunk_size=262144, shardshuffle=True)
    dl = DataLoader(
        dataset=dataset_train,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn
    )
    waveform, onset, cond = next(iter(dl))
