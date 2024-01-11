# import numpy as np
import pytorch_lightning as pl
# import importlib
from torch.utils.data import DataLoader

# from CondFoleyGen.SpecVQGAN.utils import instantiate_from_config
from CondFoleyGen.specvqgan.data.dataset_greatesthits import GreatestHitsWaveDataset, CondGreatestHitsWaveCondOnImage


class GreatestHitsWaveDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_split_file_path,
        train_data_to_use,
        val_split_file_path,
        val_data_to_use,
        test_split_file_path,
        test_data_to_use,
        chunk_length_in_seconds,
        sample_rate,
        rand_shift,
        rand_shift_range,
        audio_file_suffix,
        annotations_file_suffix,
        metadata_file_suffix,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.train_split_file_path = train_split_file_path
        self.train_data_to_use = train_data_to_use
        self.val_split_file_path = val_split_file_path
        self.val_data_to_use = val_data_to_use
        self.test_split_file_path = test_split_file_path
        self.test_data_to_use = test_data_to_use
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.sample_rate = sample_rate
        self.rand_shift = rand_shift
        self.rand_shift_range = rand_shift_range
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
            self.train_dataset = GreatestHitsWaveDataset(
                root_dir=self.root_dir,
                split_file_path=self.train_split_file_path,
                split='train',
                data_to_use=self.train_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                sample_rate=self.sample_rate,
                rand_shift=self.rand_shift,
                rand_shift_range=self.rand_shift_range,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
            )

            self.val_dataset = GreatestHitsWaveDataset(
                root_dir=self.root_dir,
                split_file_path=self.val_split_file_path,
                split='val',
                data_to_use=self.val_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                sample_rate=self.sample_rate,
                rand_shift=self.rand_shift,
                rand_shift_range=self.rand_shift_range,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
            )

            self.train_dataset.print()
            self.val_dataset.print()

        if stage == "test":
            self.test_dataset = GreatestHitsWaveDataset(
                root_dir=self.root_dir,
                split_file_path=self.test_split_file_path,
                split='test',
                data_to_use=self.test_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                sample_rate=self.sample_rate,
                rand_shift=self.rand_shift,
                rand_shift_range=self.rand_shift_range,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
            )

            self.test_dataset.print()

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


class CondGreatestHitsWaveCondOnImageDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        train_split_file_path,
        train_data_to_use,
        train_frame_transforms,

        val_split_file_path,
        val_data_to_use,
        val_frame_transforms,
        
        test_split_file_path,
        test_data_to_use,
        test_frame_transforms,
        
        chunk_length_in_seconds,
        p_outside_cond,
        p_audio_aug,
        rand_shift,
        rand_shift_range,
        sample_rate,
        audio_file_suffix,
        annotations_file_suffix,
        metadata_file_suffix,
        frame_file_suffix,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.train_split_file_path = train_split_file_path
        self.train_data_to_use = train_data_to_use
        self.train_frame_transforms = train_frame_transforms
        self.val_split_file_path = val_split_file_path
        self.val_data_to_use = val_data_to_use
        self.val_frame_transforms = val_frame_transforms
        self.test_split_file_path = test_split_file_path
        self.test_data_to_use = test_data_to_use
        self.test_frame_transforms = test_frame_transforms
        self.chunk_length_in_seconds = chunk_length_in_seconds
        self.p_outside_cond = p_outside_cond
        self.p_audio_aug = p_audio_aug
        self.rand_shift = rand_shift
        self.rand_shift_range = rand_shift_range
        self.sample_rate = sample_rate
        self.audio_file_suffix = audio_file_suffix
        self.annotations_file_suffix = annotations_file_suffix
        self.metadata_file_suffix = metadata_file_suffix
        self.frame_file_suffix = frame_file_suffix
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
            self.train_dataset = CondGreatestHitsWaveCondOnImage(
                root_dir=self.root_dir,
                split_file_path=self.train_split_file_path,
                split='train',
                data_to_use=self.train_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                frame_transforms=self.train_frame_transforms,
                p_outside_cond=self.p_outside_cond,
                p_audio_aug=self.p_audio_aug,
                rand_shift=self.rand_shift,
                rand_shift_range=self.rand_shift_range,
                sample_rate=self.sample_rate,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
            )

            self.val_dataset = CondGreatestHitsWaveCondOnImage(
                root_dir=self.root_dir,
                split_file_path=self.val_split_file_path,
                split='val',
                data_to_use=self.val_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                frame_transforms=self.val_frame_transforms,
                p_outside_cond=self.p_outside_cond,
                p_audio_aug=self.p_audio_aug,
                rand_shift=self.rand_shift,
                rand_shift_range=self.rand_shift_range,
                sample_rate=self.sample_rate,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
            )

            self.train_dataset.print()
            self.val_dataset.print()

        if stage == "test":
            self.test_dataset = CondGreatestHitsWaveCondOnImage(
                root_dir=self.root_dir,
                split_file_path=self.test_split_file_path,
                split='test',
                data_to_use=self.test_data_to_use,
                chunk_length_in_seconds=self.chunk_length_in_seconds,
                frame_transforms=self.test_frame_transforms,
                p_outside_cond=self.p_outside_cond,
                p_audio_aug=self.p_audio_aug,
                rand_shift=self.rand_shift,
                rand_shift_range=self.rand_shift_range,
                sample_rate=self.sample_rate,
                audio_file_suffix=self.audio_file_suffix,
                annotations_file_suffix=self.annotations_file_suffix,
                metadata_file_suffix=self.metadata_file_suffix,
                frame_file_suffix=self.frame_file_suffix,
            )

            self.test_dataset.print()

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
    from CondFoleyGen.specvqgan.data.transforms import *

    datamodule = GreatestHitsWaveDatamodule(
        root_dir="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed",
        train_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/train.txt",
        train_data_to_use=1.0,
        val_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/val.txt",
        val_data_to_use=1.0,
        test_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/test.txt",
        test_data_to_use=1.0,
        chunk_length_in_seconds=2.0,
        sample_rate=22050,
        rand_shift=True,
        rand_shift_range=[-0.5, 0.5],
        audio_file_suffix='.resampled.wav',
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        batch_size=32,
        num_workers=16,
        pin_memory=True,
    )

    datamodule.setup('fit')
    datamodule.setup('test')

    train_transforms = transforms.Compose([
        Resize3D(128),
        RandomResizedCrop3D(112, scale=(0.5, 1.0)),
        RandomHorizontalFlip3D(),
        ColorJitter3D(brightness=0.4, saturation=0.4, contrast=0.2, hue=0.1),
        ToTensor3D(),
        Normalize3D(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    valid_transforms = transforms.Compose([
        Resize3D(128),
        CenterCrop3D(112),
        ToTensor3D(),
        Normalize3D(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        Resize3D(128),
        CenterCrop3D(112),
        ToTensor3D(),
        Normalize3D(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    datamodule = CondGreatestHitsWaveCondOnImageDatamodule(
        root_dir="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed",
        train_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/train.txt",
        train_data_to_use=1.0,
        train_frame_transforms=train_transforms,
        val_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/val.txt",
        val_data_to_use=1.0,
        val_frame_transforms=valid_transforms,
        test_split_file_path="/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed/test.txt",
        test_data_to_use=1.0,
        test_frame_transforms=test_transforms,
        chunk_length_in_seconds=2.0,
        p_outside_cond=0.5,
        p_audio_aug=0.5,
        rand_shift=True,
        rand_shift_range=[-0.5, 0.5],
        sample_rate=22050,
        audio_file_suffix='.resampled.wav',
        annotations_file_suffix=".times.csv",
        metadata_file_suffix=".metadata.json",
        frame_file_suffix=".jpg",
        batch_size=32,
        num_workers=16,
        pin_memory=True,
    )

    datamodule.setup('fit')
    datamodule.setup('test')
