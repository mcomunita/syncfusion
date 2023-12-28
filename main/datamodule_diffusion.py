import pytorch_lightning as pl
import multiprocessing as mp
from torch.utils.data import DataLoader
from typing import List, Optional, Callable


class WebDatasetDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle_size: int,
        collate_fn: Callable
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_size = shuffle_size

        train_dataset = train_dataset.shuffle(self.shuffle_size)

        # This should help avoiding memory explosion with num_workers>0
        self.shared_data = mp.Manager().Namespace()
        self.shared_data.train_dataset = train_dataset
        self.shared_data.val_dataset = val_dataset
        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.shared_data.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.shared_data.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn
        )
