import pandas as pd
import torch
import os
import json
import random
import soundfile
import torchaudio
import librosa
import copy
import numpy as np

from PIL import Image
from torchvision import transforms
from CondFoleyGen.specvqgan.data.transforms import \
    MakeMono, \
    Padding, \
    Resize3D, \
    CenterCrop3D, \
    RandomResizedCrop3D, \
    RandomHorizontalFlip3D, \
    ColorJitter3D, \
    ToTensor3D, \
    Normalize3D


def non_negative(x): return int(np.round(max(0, x), 0))


def non_negative_time(x): return max(0, x)


def load_audio(filepath, sample_rate, frame_offset=0, num_frames=-1):
    x, sr = torchaudio.load(filepath, frame_offset, num_frames, normalize=True, channels_first=True)
    if sr != sample_rate:
        x = torchaudio.functional.resample(x, sr, sample_rate)
    return x, sr

def load_audio_librosa(filepath, sample_rate, offset_in_seconds=0, duration_in_seconds=-1):
    x, sr = librosa.load(filepath, sr=sample_rate, offset=offset_in_seconds, duration=duration_in_seconds, mono=True, dtype=np.float32)
    return x, sr


########################################################
# GREATEST HITS WAVE
########################################################
class GreatestHitsWaveDataset(torch.utils.data.Dataset):
    """
    Dataset of audio files extracted from videos in Greatest Hits dataset.
    A list of all videos in the dataset is created together with a list of
    [video, onset time, video duration] from annotations.
    Class returns a chunk of audio of length L seconds around the onset times.
    Random shift is applied before extracting audio chunks.

    item = {
        'image': audio, # (L * sample_rate)
        'file_path_wav_': audio_path,
        'label': None,
        'target': None
    }
    """

    def __init__(
        self,
        root_dir,
        split_file_path,
        split='train',
        data_to_use=1.0,
        chunk_length_in_seconds=2.0,
        sample_rate=22050,
        rand_shift=True,
        rand_shift_range=[-0.5, 0.5],
        audio_file_suffix='.resampled.wav',
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json"
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_file_path = split_file_path
        self.split = split
        self.data_to_use = data_to_use
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.sample_rate = sample_rate
        self.rand_shift = rand_shift
        self.rand_shift_range = rand_shift_range
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.list_samples = []  # list of video/audio names
        self.list_onsets = []  # list of [video, onset time] tuples extracted from annotations

        # get list of video names
        with open(split_file_path, "r") as f:
            self.list_samples = f.read().splitlines()

        # subset of videos
        if data_to_use < 1.0:
            # shuffle list
            random.shuffle(self.list_samples)
            self.list_samples = self.list_samples[0:int(len(self.list_samples) * data_to_use)]
            # reorder
            self.list_samples.sort()

        # # create list of [video, onset time] pairs
        # for sample in self.list_samples:
        #     # get annotations
        #     annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
        #     annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
        #     onset_times = annotations["times"].values

        #     # add to list
        #     for onset_time in onset_times:
        #         self.list_onsets.append([sample, onset_time])

        # create list of [video name, onset time, video duration] tuples
        for sample in self.list_samples:
            # get annotations
            annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
            annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
            onset_times = annotations["times"].values
            # get metadata
            metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            duration = metadata["processed"]["video_duration"]

            # add to list
            for onset_time in onset_times:
                self.list_onsets.append([sample, onset_time, duration])

        # get sample rate from a sample
        # sample = self.list_samples[0]
        # metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
        # with open(metadata_path, "r") as f:
        #     metadata = json.load(f)
        # self.sample_rate = metadata["processed"]["audio_sample_rate"]

        # audio transform
        self.audio_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(self.sample_rate * self.chunk_length_in_seconds)),
        ])

    def __len__(self):
        return len(self.list_onsets)

    def __getitem__(self, idx):
        sample, onset_time, duration = self.list_onsets[idx]
        
        # start and end time for audio chunk
        start_time = onset_time
        if self.rand_shift:
            shift = random.uniform(self.rand_shift_range[0], self.rand_shift_range[1])
            start_time = max(start_time + shift, 0)  # non negative start
        start_time = min(start_time, duration - self.chunk_length_in_seconds)  # make sure not to exceed duration
        
        # load audio
        audio_path = os.path.join(self.root_dir, sample, "audio", f"{sample}{self.audio_file_suffix}")
        # audio, sr = soundfile.read(
        #     audio_path,
        #     frames=int(self.sample_rate * self.chunk_length_in_seconds),
        #     start=start_idx,
        #     samplerate=self.sample_rate,
        # )
        # audio, sr = load_audio(
        #     audio_path,
        #     sample_rate=self.sample_rate,
        #     frame_offset=start_idx,
        #     num_frames=int(self.sample_rate * self.chunk_length_in_seconds)
        # )
        audio, sr = load_audio_librosa(
            audio_path,
            sample_rate=self.sample_rate,
            offset_in_seconds=start_time,
            duration_in_seconds=self.chunk_length_in_seconds,
        )    
        audio = self.audio_transforms(audio)

        item = {
            'image': audio,
            'file_path_wav_': audio_path,
            # 'label': None,
            # 'target': None
        }

        return item

    def print(self):
        print(f"\nGreatest Hits Wave {self.split} dataset:")
        print(f"num {self.split} samples: {len(self.list_samples)}")
        print(f"num {self.split} onsets: {len(self.list_onsets)}")
        example = self[0]
        print(f"example['image']: {example['image'].shape}")
        print(f"example['file_path_wav_']: {example['file_path_wav_']}")


########################################################
# GREATEST HITS WAVE cond on IMAGE
########################################################
class CondGreatestHitsWaveCondOnImage(torch.utils.data.Dataset):
    """
    Dataset of audio and video files extracted from videos in Greatest Hits dataset.
    Conditioned on audio and video chunks from same or different video.
    A list of all videos in the dataset is created together with a list of
    [video, onset time, video duration] from annotations.
    Class returns audio and video chunks of length L seconds around the onset times,
    together with conditioning audio and video chunks of length L seconds from same or
    different video.
    Random shift is applied before extracting audio chunks.

    item = {
            "image": audio,
            "cond_image": cond_audio,
            "file_path_wav_": audio_path,
            "file_path_cond_wav_": cond_audio_path,
            "feature": np.stack(cond_frames + frames, axis=0),
            "file_path_feats_": (frames_path, start_video_frame),
            "file_path_cond_feats_": (cond_frames_path, cond_start_video_frame),
            "label": None,
            "target": None,
        }
    """
    def __init__(
        self,
        root_dir,
        split_file_path,
        split='train',
        data_to_use=1.0,
        chunk_length_in_seconds=2.0,
        frame_transforms=None,
        p_outside_cond=0.,
        p_audio_aug=0.5,
        rand_shift=True,
        rand_shift_range=[-0.5, 0.5],
        sample_rate=22050,
        audio_file_suffix=".resampled.wav",
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_file_path = split_file_path
        self.split = split
        self.data_to_use = data_to_use
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.frame_transforms = frame_transforms
        self.p_outside_cond = p_outside_cond
        self.p_audio_aug = p_audio_aug
        self.rand_shift = rand_shift
        self.rand_shift_range = rand_shift_range
        self.sample_rate = sample_rate
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.list_samples = []  # list of video/audio names
        self.list_onsets = []  # list of [video, onset time] tuples extracted from annotations

        # get list of video names
        with open(split_file_path, "r") as f:
            self.list_samples = f.read().splitlines()

        # subset of videos
        if data_to_use < 1.0:
            # shuffle list
            random.shuffle(self.list_samples)
            self.list_samples = self.list_samples[0:int(len(self.list_samples) * data_to_use)]
            self.list_samples.sort()

        # create list of [video name, onset time, video duration] tuples
        for sample in self.list_samples:
            # get annotations
            annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
            annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
            onset_times = annotations["times"].values
            # get metadata
            metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            duration = metadata["processed"]["video_duration"]

            # add to list
            for onset_time in onset_times:
                self.list_onsets.append([sample, onset_time, duration])
        
        # create dictionary of {"video_name": indeces of onsets belonging to video_name} pairs
        self.dict_video_onsets = {}
        for i, (sample, onset_time, duration) in enumerate(self.list_onsets):
            if sample not in self.dict_video_onsets:
                self.dict_video_onsets[sample] = []
            self.dict_video_onsets[sample].append(i)

        # get frame rate from a sample
        sample = self.list_samples[0]
        metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.frame_rate = metadata["processed"]["video_frame_rate"]

        # audio transform
        self.audio_transforms = transforms.Compose([
            MakeMono(),
            Padding(target_len=int(self.sample_rate * self.chunk_length_in_seconds)),
        ])

        # if frame transforms not specified, use default
        if self.frame_transforms == None:
            self.frame_transforms = transforms.Compose([
                Resize3D(128),
                CenterCrop3D(112),
                ToTensor3D(),
                Normalize3D(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.list_onsets)

    def __getitem__(self, idx):
        sample, onset_time, duration = self.list_onsets[idx]

        # -- reference audio/video
        # start and end frames for audio/video chunks
        start_time = onset_time
        if self.rand_shift:
            shift = random.uniform(self.rand_shift_range[0], self.rand_shift_range[1])
            start_time = max(start_time + shift, 0)  # non negative start
        start_time = min(start_time, duration - self.chunk_length_in_seconds) # make sure not to exceed duration
        end_time = start_time + self.chunk_length_in_seconds
        # start_audio_frame = int(start_time * self.sample_rate)
        start_video_frame = int(start_time * self.frame_rate)
        end_video_frame = int(end_time * self.frame_rate)
        
        # load audio chunk
        audio_path = os.path.join(self.root_dir, sample, "audio", f"{sample}{self.audio_file_suffix}")
        audio, sr = load_audio_librosa(
            audio_path,
            sample_rate=self.sample_rate,
            offset_in_seconds=start_time,
            duration_in_seconds=self.chunk_length_in_seconds,
        )  

        # load video chunk
        frames_path = os.path.join(self.root_dir, sample, "frames")
        frames = [
            Image.open(os.path.join(frames_path, f'{sample}.frame_{i+1:0>6d}{self.frame_file_suffix}')
                       ).convert('RGB') for i in range(start_video_frame, end_video_frame)
        ]

        # -- conditioning audio/video
        # use cond. chunk from different video of ref chunk
        if torch.bernoulli(torch.tensor(self.p_outside_cond)) == 1.:
            cond_idx = random.randint(0, len(self)-1)
            cond_sample, cond_onset_time, cond_duration = self.list_onsets[cond_idx]
            while cond_sample == sample:
                cond_idx = random.randint(0, len(self)-1)
                cond_sample, cond_onset_time, cond_duration = self.list_onsets[cond_idx]
        # use cond. chunk from same video as ref chunk
        else:
            cond_sample = sample
            onsets_idxs = copy.copy(self.dict_video_onsets[cond_sample])
            onsets_idxs.remove(idx) # remove idx of ref chunk
            cond_idx = random.sample(onsets_idxs, k=1)[0]
            cond_sample, cond_onset_time, cond_duration = self.list_onsets[cond_idx]
        
        # start and end frames for audio/video chunks
        cond_start_time = cond_onset_time
        if self.rand_shift:
            shift = random.uniform(self.rand_shift_range[0], self.rand_shift_range[1])
            cond_start_time = max(cond_start_time + shift, 0)  # non negative start
        cond_start_time = min(cond_start_time, cond_duration - self.chunk_length_in_seconds)  # make sure not to exceed duration
        cond_end_time = cond_start_time + self.chunk_length_in_seconds
        cond_start_video_frame = int(cond_start_time * self.frame_rate)
        cond_end_video_frame = int(cond_end_time * self.frame_rate)

        # load audio chunk
        cond_audio_path = os.path.join(self.root_dir, cond_sample, "audio", f"{cond_sample}{self.audio_file_suffix}")
        cond_audio, sr = load_audio_librosa(
            cond_audio_path,
            sample_rate=self.sample_rate,
            offset_in_seconds=cond_start_time,
            duration_in_seconds=self.chunk_length_in_seconds,
        )

        # load video chunk
        cond_frames_path = os.path.join(self.root_dir, cond_sample, "frames")
        cond_frames = [
            Image.open(os.path.join(cond_frames_path, f'{cond_sample}.frame_{i+1:0>6d}{self.frame_file_suffix}')
                       ).convert('RGB') for i in range(cond_start_video_frame, cond_end_video_frame)
        ]

        # apply audio transforms
        audio = self.audio_transforms(audio)
        cond_audio = self.audio_transforms(cond_audio)

        # apply video transforms
        if self.frame_transforms is not None:
            frames = self.frame_transforms(frames)
            cond_frames = self.frame_transforms(cond_frames)

        item = {
            "image": audio,
            "cond_image": cond_audio,
            "file_path_wav_": audio_path,
            "file_path_cond_wav_": cond_audio_path,
            # "feature": np.stack(cond_frames + frames, axis=0),
            "feature": torch.stack(cond_frames + frames, axis=0),
            "file_path_feats_": (frames_path, start_video_frame),
            "file_path_cond_feats_": (cond_frames_path, cond_start_video_frame),
            # "label": None,
            # "target": None,
        }

        return item

    def print(self):
        print(f"\nCond. Greatest Hits Wave Cond. on Image {self.split} dataset:")
        print(f"num {self.split} samples: {len(self.list_samples)}")
        print(f"num {self.split} onsets: {len(self.list_onsets)}")
        example = self[0]
        print(f"example['image']: {example['image'].shape}")
        print(f"example['cond_image']: {example['cond_image'].shape}")
        print(f"example['file_path_wav_']: {example['file_path_wav_']}")
        print(f"example['file_path_cond_wav_']: {example['file_path_cond_wav_']}")
        print(f"example['feature']: {example['feature'].shape}")
        print(f"example['file_path_feats_']: {example['file_path_feats_']}")
        print(f"example['file_path_cond_feats_']: {example['file_path_cond_feats_']}")


if __name__ == '__main__':
    dataset = GreatestHitsWaveDataset(
        root_dir="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed",
        split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/train.txt",
        split='train',
        data_to_use=1.0,
        chunk_length_in_seconds=2.0,
        sample_rate=22050,
        rand_shift=True,
        rand_shift_range=[-0.5, 0.5],
        audio_file_suffix='.resampled.wav',
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json"
    )
    dataset.print()