import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Union
from main.dataset_onset import GreatestHitsDataset

############################################################################
# DATA MODULE
##############################################################################

class GreatestHitsDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        train_split_file_path: str,
        train_data_to_use: float,
        train_frames_transforms: Union[transforms.Compose, None],
        val_split_file_path: str,
        val_data_to_use: float,
        val_frames_transforms: Union[transforms.Compose, None],
        test_split_file_path: str,
        test_data_to_use: float,
        test_frames_transforms: Union[transforms.Compose, None],
        chunk_length_in_seconds: float,
        audio_file_suffix: str,
        annotations_file_suffix: str,
        metadata_file_suffix: str,
        frame_file_suffix: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.train_split_file_path = train_split_file_path
        self.train_data_to_use = train_data_to_use
        self.train_frames_transforms = train_frames_transforms
        self.val_split_file_path = val_split_file_path
        self.val_data_to_use = val_data_to_use
        self.val_frames_transforms = val_frames_transforms
        self.test_split_file_path = test_split_file_path
        self.test_data_to_use = test_data_to_use
        self.test_frames_transforms = test_frames_transforms
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
            self.train_dataset = GreatestHitsDataset(
                root_dir=self.root_dir,
                split_file_path=self.train_split_file_path,
                split='train',
                data_to_use=self.train_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                frames_transforms=self.train_frames_transforms,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
            )

            self.val_dataset = GreatestHitsDataset(
                root_dir=self.root_dir,
                split_file_path=self.val_split_file_path,
                split='val',
                data_to_use=self.val_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                frames_transforms=self.val_frames_transforms,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
            )

            self.train_dataset.print()
            self.val_dataset.print()

        if stage == "test":
            self.test_dataset = GreatestHitsDataset(
                root_dir=self.root_dir,
                split_file_path=self.test_split_file_path,
                split='test',
                data_to_use=self.test_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                frames_transforms=self.test_frames_transforms,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


if __name__ == '__main__':
    from torchvision import transforms

    train_transforms = [
        transforms.Resize((128, 128), antialias=True),
        transforms.RandomCrop((112, 112)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    val_transforms = [
        transforms.Resize((128, 128), antialias=True),
        transforms.RandomCrop((112, 112)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    test_transforms = [
        transforms.Resize((128, 128), antialias=True),
        transforms.CenterCrop((112, 112)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


    datamodule = GreatestHitsDatamodule(
        root_dir="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed",
        train_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/train.txt",
        train_data_to_use=1.0,
        train_frames_transforms=train_transforms,

        val_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/val.txt",
        val_data_to_use=1.0,
        val_frames_transforms=val_transforms,

        test_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/test.txt",
        test_data_to_use=1.0,
        test_frames_transforms=test_transforms,

        chunk_length_in_seconds=2.0,

        audio_file_suffix=".resampled.wav",
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
        
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )