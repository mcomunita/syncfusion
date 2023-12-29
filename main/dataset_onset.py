import torch
import os
import json
import pandas as pd
import random
import glob
import torchvision.transforms as transforms
from natsort import natsorted

from PIL import Image


class GreatestHitsDataset(torch.utils.data.Dataset):
    """
    Dataset to train onset detection model on Greatest Hits dataset. 
    Split videos into chunks of N seconds.
    Annotate each chunk with onset labels (1 if onset, 0 otherwise) for each video frame.
    """

    def __init__(
        self,
        root_dir,
        split_file_path,
        split='train',
        data_to_use=1.0,
        chunk_length_in_seconds=2.0,
        frames_transforms=None,
        audio_file_suffix='.resampled.wav',
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
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.frame_file_suffix = frame_file_suffix

        if frames_transforms is not None:
            self.frames_transforms = frames_transforms
        else:
            self.frames_transforms = transforms.Compose([
                transforms.Resize((112, 112), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # read list of samples
        with open(split_file_path, "r") as f:
            self.list_samples = f.read().splitlines()

        # subset
        if data_to_use < 1.0:
            # shuffle list
            random.shuffle(self.list_samples)
            self.list_samples = self.list_samples[0:int(len(self.list_samples) * data_to_use)]
            self.list_samples = natsorted(self.list_samples)  # natural sorting (e.g. 1, 2, 10 instead of 1, 10, 2)

        self.list_chunks = []
        self.total_time_in_minutes = 0.0

        for sample in self.list_samples:
            # get metadata
            metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            frame_rate = metadata["processed"]["video_frame_rate"]
            duration = metadata["processed"]["video_duration"]

            # compute number of chunks
            num_chunks = int(duration / chunk_length_in_seconds)
            end_time = num_chunks * chunk_length_in_seconds

            # get annotations (onsets) up to end time
            annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
            annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
            onset_times = annotations["times"].values
            onset_times = onset_times[onset_times < end_time]

            self.total_time_in_minutes += end_time

            # get chunks
            chunk_length_in_frames = int(chunk_length_in_seconds * frame_rate)
            for i in range(num_chunks):
                # chunk start and end
                chunk_start_time = i * chunk_length_in_seconds
                chunk_end_time = chunk_start_time + chunk_length_in_seconds
                chunk_start_frame = int(chunk_start_time * frame_rate)
                chunk_end_frame = int(chunk_end_time * frame_rate)

                # extract onset times for this chunk
                chunk_onsets_times = annotations[(annotations["times"] >= chunk_start_time) & (annotations["times"] < chunk_end_time)]["times"].values

                # normalize to chunk start
                chunk_onsets_times = chunk_onsets_times - chunk_start_time

                # convert onset times to frames
                chunk_onsets_frames = (chunk_onsets_times * frame_rate).astype(int)

                # compute frames labels
                labels = torch.zeros(chunk_length_in_frames)
                labels[chunk_onsets_frames] = 1

                # append chunk
                self.list_chunks.append({
                    "video_name": sample,
                    "frames_path": os.path.join(root_dir, sample, "frames"),
                    "start_time": chunk_start_time,
                    "end_time": chunk_end_time,
                    "start_frame": chunk_start_frame,
                    "end_frame": chunk_end_frame,
                    "labels": labels,
                    "frame_rate": frame_rate

                })

        self.total_time_in_minutes /= 60.0

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, index):

        chunk = self.list_chunks[index]
        frames_list = glob.glob(f"{chunk['frames_path']}/*{self.frame_file_suffix}")
        frames_list = natsorted(frames_list)

        # get frames
        frames_list = frames_list[chunk["start_frame"]:chunk["end_frame"]]
        imgs = self.read_image_and_apply_transforms(frames_list)

        # get labels
        labels = chunk["labels"]

        item = {
            "video_name": chunk["video_name"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "start_frame": chunk["start_frame"],
            "end_frame": chunk["end_frame"],
            "frames": imgs,
            "label": labels,
            "frame_rate": chunk["frame_rate"],
        }

        return item

    def read_image_and_apply_transforms(self, frame_list):
        imgs = []
        convert_tensor = transforms.ToTensor()
        for img_path in frame_list:
            image = Image.open(img_path).convert('RGB')
            image = convert_tensor(image)
            imgs.append(image.unsqueeze(0))
        # (T, C, H ,W)
        imgs = torch.cat(imgs, dim=0).squeeze()
        if self.frames_transforms is not None:
            imgs = self.frames_transforms(imgs)
        imgs = imgs.permute(1, 0, 2, 3)
        # (C, T, H ,W)
        return imgs

    def print(self):
        print(f"\nGreatesthit {self.split} dataset:")
        print(f"num {self.split} samples: {len(self.list_samples)}")
        print(f"num {self.split} chunks: {len(self.list_chunks)}")
        print(f"total time in minutes: {self.total_time_in_minutes}")
        print(f"chunk frames size: {self[0]['frames'].shape}")
        print(f"chunk label size: {self[0]['label'].shape}")


if __name__ == '__main__':
    dataset = GreatestHitsDataset(
        root_dir="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed",
        split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/train.txt",
        split='train',
        data_to_use=1.0,
        chunk_length_in_seconds=2.0,
        frames_transforms=[
            transforms.Resize((128, 128), antialias=True),
            transforms.RandomCrop((112, 112)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        audio_file_suffix=".resampled.wav",
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
    )
    dataset.print()


# class GreatestHitsDataset(torch.utils.data.Dataset):
#     """
#     Dataset to train onset detection model on Greatest Hits dataset.
#     Split videos into chunks of N seconds (chunks start at integer second before first onset).
#     Annotate each chunk with onset labels (1 if onset, 0 otherwise) for each video frame
#     """

#     def __init__(
#         self,
#         root_dir,
#         split_file_path,
#         split='train',
#         data_to_use=1.0,
#         chunk_length_in_seconds=2.0,
#         audio_file_suffix='.resampled.wav',
#         annotations_file_suffix=".times.csv",
#         metadata_file_suffix=".metadata.json"
#     ):
#         super().__init__()
#         self.root_dir = root_dir
#         self.split_file_path = split_file_path
#         self.split = split
#         self.data_to_use = data_to_use
#         self.chunk_length_in_seconds = chunk_length_in_seconds
#         self.audio_file_suffix = audio_file_suffix
#         self.annotations_file_suffix = annotations_file_suffix
#         self.metadata_file_suffix = metadata_file_suffix

#         self.video_transform = transforms.Compose(
#             self.generate_video_transform()
#         )

#         with open(split_file_path, "r") as f:
#             self.list_samples = f.read().splitlines()

#         if data_to_use < 1.0:
#             # shuffle list
#             random.shuffle(self.list_samples)
#             self.list_samples = self.list_samples[0:int(len(self.list_samples) * data_to_use)]

#         self.list_chunks = []
#         self.total_time_in_minutes = 0.0

#         for sample in self.list_samples:
#             # get metadata
#             metadata_path = os.path.join(root_dir, sample, f"{sample}{metadata_file_suffix}")
#             with open(metadata_path, "r") as f:
#                 metadata = json.load(f)
#             frame_rate = metadata["processed"]["video_frame_rate"]

#             # get annotations
#             annotations_path = os.path.join(root_dir, sample, f"{sample}{annotations_file_suffix}")
#             annotations = pd.read_csv(annotations_path, header=None, names=["times", "labels"])
#             onset_times = annotations["times"].values
#             start_time = int(onset_times[0])  # start at integer second before first hit
#             end_time = onset_times[-1] - (onset_times[-1] - start_time) % chunk_length_in_seconds

#             self.total_time_in_minutes += end_time - start_time

#             # get chunks
#             num_chunks = int((end_time - start_time) / chunk_length_in_seconds)
#             chunk_length_in_frames = int(chunk_length_in_seconds * frame_rate)
#             for i in range(num_chunks):
#                 # chunk start and end
#                 chunk_start_time = start_time + i * chunk_length_in_seconds
#                 chunk_end_time = chunk_start_time + chunk_length_in_seconds
#                 chunk_start_frame = int(chunk_start_time * frame_rate)
#                 chunk_end_frame = int(chunk_end_time * frame_rate)

#                 # extract onset times for this chunk
#                 chunk_onsets_labels = annotations[(annotations["times"] >= chunk_start_time) & (annotations["times"] < chunk_end_time)]["times"].values

#                 # normalize to chunk start
#                 chunk_onsets_labels = chunk_onsets_labels - chunk_start_time

#                 # convert onset times to frames
#                 chunk_onsets_frames = (chunk_onsets_labels * frame_rate).astype(int)

#                 # compute frames labels
#                 labels = torch.zeros(chunk_length_in_frames)
#                 labels[chunk_onsets_frames] = 1

#                 # append chunk
#                 self.list_chunks.append({
#                     "frames_path": os.path.join(root_dir, sample, "frames"),
#                     "start_frame": chunk_start_frame,
#                     "end_frame": chunk_end_frame,
#                     "labels": labels
#                 })

#         self.total_time_in_minutes /= 60.0

#     def __getitem__(self, index):

#         chunk = self.list_chunks[index]
#         frames_list = glob.glob(f"{chunk['frames_path']}/*.jpg")
#         frames_list.sort()

#         # get frames
#         frames_list = frames_list[chunk["start_frame"]:chunk["end_frame"]]
#         imgs = self.read_image(frames_list)

#         # get labels
#         labels = chunk["labels"]

#         batch = {
#             'frames': imgs,
#             'label': labels
#         }

#         return batch

#     def __len__(self):
#         return len(self.list_chunks)

#     def read_image(self, frame_list):
#         imgs = []
#         convert_tensor = transforms.ToTensor()
#         for img_path in frame_list:
#             image = Image.open(img_path).convert('RGB')
#             image = convert_tensor(image)
#             imgs.append(image.unsqueeze(0))
#         # (T, C, H ,W)
#         imgs = torch.cat(imgs, dim=0).squeeze()
#         imgs = self.video_transform(imgs)
#         imgs = imgs.permute(1, 0, 2, 3)
#         # (C, T, H ,W)
#         return imgs

#     def generate_video_transform(self):
#         vision_transform_list = []

#         vision_transform_list.append(transforms.Resize((128, 128), antialias=True))

#         if self.split == 'train':
#             vision_transform_list.append(transforms.RandomCrop((112, 112)))
#             vision_transform_list.append(transforms.ColorJitter(
#                 brightness=0.1, contrast=0.1, saturation=0, hue=0
#             ))
#         else:
#             vision_transform_list.append(transforms.CenterCrop((112, 112)))
#             # color_funct = transforms.Lambda(lambda img: img)

#         vision_transform_list.append(transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#         ))

#         # vision_transform_list = [
#         #     resize_funct,
#         #     crop_funct,
#         #     color_funct,
#         #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         # ]

#         return vision_transform_list

#     def print(self):
#         print(f"\nGreatesthit {self.split} dataset:")
#         print(f"num {self.split} samples: {len(self.list_samples)}")
#         print(f"num {self.split} chunks: {len(self.list_chunks)}")
#         print(f"total time in minutes: {self.total_time_in_minutes}")
#         print(f"chunk frames size: {self[0]['frames'].shape}")
#         print(f"chunk label size: {self[0]['label'].shape}")
