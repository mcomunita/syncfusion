import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import wandb
import numpy as np
import glob
import pytorch_lightning as pl

from sklearn.metrics import average_precision_score
from natsort import natsorted
from torch.utils.data import DataLoader
from main.dataset_onset import GreatestHitsDataset
from main.onset_net import VideoOnsetNet


##############################################################################
# MODEL
##############################################################################

class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        onset_model: VideoOnsetNet
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.model: VideoOnsetNet = onset_model
        self.loss = BCLoss()
        # self.sanity_check_done = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    # def on_train_start(self):
    #     self.sanity_check_done = True

    # def on_test_start(self):
    #     self.sanity_check_done = True

    def common_step(self, batch, batch_idx, mode="train"):
        frames, labels = batch['frames'], batch['label']
        pred = self.model(frames)
        loss = self.loss(pred, labels)

        # print("\ncommon step")
        # print(f"pred: {pred.shape}")
        # print(f"labels: {labels.shape}")

        # if self.sanity_check_done:
        metrics = self.loss.evaluate(pred, labels)
        return loss, metrics
        # return loss, None

    def training_step(self, batch, batch_idx):
        loss, metrics = self.common_step(batch, batch_idx, mode="train")

        # log loss
        self.log(
            f"loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # log metrics
        # if self.sanity_check_done > 0:
        self.log_metrics(metrics, mode="train")

        # if not self.sanity_check_done:
        #     self.sanity_check_done = True

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.common_step(batch, batch_idx, mode="val")
        self.log(
            f"loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # if self.sanity_check_done > 0:
        # log metrics
        self.log_metrics(metrics, mode="val")

        # log labels
        if batch_idx == 0:
            frames, labels = batch['frames'], batch['label']
            pred = self.model(frames)

            self.log_labels(batch_idx, batch, pred, mode="val")

        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.common_step(batch, batch_idx, mode="test")

        # log loss
        self.log(
            f"loss/test",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # log metrics
        self.log_metrics(metrics, mode="test")

        self.log_annotations(batch_idx, batch, mode="test")

        # log labels
        if batch_idx % 10 == 0:
            frames, labels = batch['frames'], batch['label']
            pred = self.model(frames)

            self.log_labels(batch_idx, batch, pred, mode="test")

        return loss

    def on_test_epoch_end(self):
        self.concat_annotations()

    def log_metrics(self, metrics, mode="val"):
        for key, value in metrics.items():
            self.log(
                f"metrics/{mode}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def log_annotations(self, batch_idx, batch, mode="test"):
        video_name = batch['video_name']
        start_frame = batch['start_frame']
        end_frame = batch['end_frame']
        frame_rate = batch['frame_rate']
        frames = batch['frames']
        target_labels = batch['label'].cpu().numpy()

        # output paths
        run_dir = self.logger.experiment.dir
        target_dir = os.path.join(run_dir, f"media/annotations/target")
        pred_dir = os.path.join(run_dir, f"media/annotations/pred")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        # get predictions
        pred = self.model(frames)
        pred_labels = (pred > 0.5).float().cpu().numpy()

        # print("log annotations")
        # print(f"pred: {pred_labels.shape}")
        # print(f"labels: {target_labels.shape}")

        for i, _ in enumerate(video_name):
            # get indeces of onsets by row
            target_indecies = np.nonzero(target_labels[i])[0]
            pred_indecies = np.nonzero(pred_labels[i])[0]

            # remove consecutive onsets in pred
            for j in range(len(pred_indecies) - 2):
                if pred_indecies[j] == 1 and pred_indecies[j + 1] == 1:
                    pred_indecies = np.delete(pred_indecies, j + 1)

            # get the corresponding time stamps
            target_times = (target_indecies + start_frame[i].cpu().numpy()) / frame_rate[i].cpu().numpy()
            pred_times = (pred_indecies + start_frame[i].cpu().numpy()) / frame_rate[i].cpu().numpy()

            # save annotations to csv
            target_file_path = os.path.join(target_dir, f"{video_name[i]}.{start_frame[i]}-{end_frame[i]}.times.csv")
            np.savetxt(target_file_path, target_times, fmt="%.4f", delimiter=",")

            pred_file_path = os.path.join(pred_dir, f"{video_name[i]}.{start_frame[i]}-{end_frame[i]}.times.csv")
            np.savetxt(pred_file_path, pred_times, fmt="%.4f", delimiter=",")

    def concat_annotations(self):
        # merge content of all files with the same video name
        # and save to a single file
        run_dir = self.logger.experiment.dir
        target_dir = os.path.join(run_dir, f"media/annotations/target")
        pred_dir = os.path.join(run_dir, f"media/annotations/pred")

        target_file_paths = natsorted(glob.glob(os.path.join(target_dir, "*.times.csv")))
        video_names = set([os.path.basename(target_file_path).split(".")[0] for target_file_path in target_file_paths])

        for video_name in video_names:
            # target file
            video_file_paths = natsorted(glob.glob(os.path.join(target_dir, f"{video_name}.*.times.csv")))
            video_times = []
            for video_file_path in video_file_paths:
                times = np.loadtxt(video_file_path, delimiter=",").tolist()
                if isinstance(times, float):
                    times = [times]
                video_times.append(times)
            video_times = [item for sublist in video_times for item in sublist]
            video_file_path = os.path.join(target_dir, f"{video_name}.times.csv")
            np.savetxt(video_file_path, video_times, fmt="%.4f", delimiter="\n")

            # pred file
            video_file_paths = natsorted(glob.glob(os.path.join(pred_dir, f"{video_name}.*.times.csv")))
            video_times = []
            for video_file_path in video_file_paths:
                times = np.loadtxt(video_file_path, delimiter=",").tolist()
                if isinstance(times, float):
                    times = [times]
                video_times.append(times)
            video_times = [item for sublist in video_times for item in sublist]
            video_file_path = os.path.join(pred_dir, f"{video_name}.times.csv")
            np.savetxt(video_file_path, video_times, fmt="%.4f", delimiter="\n")

        # delete source files
        target_file_paths = natsorted(glob.glob(os.path.join(target_dir, "*.*.times.csv")))
        for target_file_path in target_file_paths:
            os.remove(target_file_path)

        pred_file_paths = natsorted(glob.glob(os.path.join(pred_dir, "*.*.times.csv")))
        for pred_file_path in pred_file_paths:
            os.remove(pred_file_path)

    def log_labels(self, batch_idx, batch, pred, mode="val"):
        # if self.sanity_check_done:
        video_name = batch["video_name"]
        start_time = batch["start_time"]
        end_time = batch["end_time"]
        start_frame = batch["start_frame"]
        end_frame = batch["end_frame"]
        target = batch["label"]
        pred = torch.sigmoid(pred)

        for i, _ in enumerate(target):
            v_name = video_name[i]
            s_time = start_time[i]
            e_time = end_time[i]
            s_frame = start_frame[i]
            e_frame = end_frame[i]
            target_labels = target[i].cpu().numpy()
            pred_prob = pred[i]
            pred_labels = (pred[i] > 0.5).float().cpu().numpy()

            columns = ["frame", "label"]
            target_data = [[x, y] for (x, y) in zip(list(range(1, len(target_labels) + 1)), target_labels)]
            target_table = wandb.Table(data=target_data, columns=columns)

            # pred_data = [[x, y] for (x, y) in zip(list(range(1, len(pred_labels) + 1)), pred_labels)]
            pred_data = [[x, y] for (x, y) in zip(list(range(1, len(pred_labels) + 1)), pred_prob)]
            pred_table = wandb.Table(data=pred_data, columns=columns)

            self.logger.experiment.log({
                f"labels/{mode}/target/b{batch_idx}-{i}": wandb.plot.line(target_table, x="frame", y="label", title=f"Onsets b{i} - {v_name} - time {s_time}-{e_time} - frames {s_frame}-{e_frame}"),
                f"labels/{mode}/pred/b{batch_idx}-{i}": wandb.plot.line(pred_table, x="frame", y="label", title=f"Onsets b{i} - {v_name} - time {s_time}-{e_time} - frames {s_frame}-{e_frame}"),
            }, step=self.trainer.global_step)


##############################################################################
# LOSS
##############################################################################

class BCLoss(torch.nn.Module):
    # binary classification loss
    def __init__(self):
        super(BCLoss, self).__init__()
        self.threshold = 0.75

    def forward(self, pred, target):
        # flatten
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # compute weighted BCE loss to balance positive and negative samples
        pos_weight = (target.shape[0] - target.sum()) / target.sum()

        # init weighted BCE loss
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(pred.device)

        loss = criterion(pred, target.float())
        return loss

    def evaluate(self, pred, target):
        # print("evaluate")
        # print(f"pred: {pred.shape}")
        # print(f"target: {target.shape}")

        ons_num_acc = self.onset_num_acc(pred, target)

        # flatten
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        pred = torch.sigmoid(pred)

        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()

        # print(f"pred: {pred.shape}")
        # print(f"target: {target.shape}")

        # indeces of positive and negative samples
        pos_index = np.nonzero(target == 1)[0]
        neg_index = np.nonzero(target == 0)[0]

        # balance the number of positive and negative samples
        balance_num = min(pos_index.shape[0], neg_index.shape[0])
        index = np.concatenate((pos_index[:balance_num], neg_index[:balance_num]), axis=0)

        # get the corresponding predictions and targets
        pred = pred[index]
        target = target[index]

        # compute metrics
        ap = average_precision_score(target, pred)
        acc = self.binary_acc(pred, target)
        res = {
            'AP': ap,
            'Acc': acc,
            'OnsNumAcc': ons_num_acc,
        }
        # print(res)
        return res

    def binary_acc(self, pred, target):
        # pred = pred > self.threshold
        pred[pred > self.threshold] = 1
        pred[pred <= self.threshold] = 0
        acc = np.sum(pred == target) / target.shape[0]
        # print(f"pred: {pred}")
        # print(f"target: {target}")
        # print(f"acc: {acc}")
        return acc

    def onset_num_acc(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()

        # pred = pred > self.threshold
        pred[pred > self.threshold] = 1
        pred[pred <= self.threshold] = 0

        pred = pred.astype(int)
        target = target.astype(int)

        # remove consecutive onsets in pred
        for i in range(pred.shape[0]):
            for j in range(pred.shape[-1] - 1):
                if pred[i][j] == 1 and pred[i][j + 1] == 1:
                    pred[i][j + 1] = 0

        # print(f"pred: {pred[0]}")
        # print(f"target: {target[0]}")

        pred = np.sum(pred, axis=-1)
        target = np.sum(target, axis=-1)

        # pred_onset_num = np.sum(pred)
        # target_onset_num = np.sum(target)

        acc_num_onset = np.sum(pred == target)
        # print(f"acc_num_onset: {acc_num_onset}")
        # print(f"len(pred): {len(pred)}")
        # print(f"acc_num_onset / len(pred): {acc_num_onset / len(pred)}")

        return acc_num_onset / len(pred)

        # print(f"pred_onset_num: {pred_onset_num}")
        # print(f"target_onset_num: {target_onset_num}")

        # if pred_onset_num == target_onset_num:
        #     return 1
        # else:
        #     return 0


##############################################################################
# DATA MODULE
##############################################################################

# class Datamodule(pl.LightningDataModule):
#     def __init__(
#         self,
#         root_dir,
#         train_split_file_path,
#         train_data_to_use,
#         # train_repeat,
#         # train_max_sample,
#         val_split_file_path,
#         val_data_to_use,
#         # val_repeat,
#         # val_max_sample,
#         test_split_file_path,
#         test_data_to_use,
#         # test_repeat,
#         # test_max_sample,
#         chunk_length_in_seconds,
#         audio_file_suffix,
#         annotations_file_suffix,
#         metadata_file_suffix,
#         batch_size: int,
#         num_workers: int,
#         pin_memory: bool,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.root_dir = root_dir
#         self.train_split_file_path = train_split_file_path
#         self.train_data_to_use = train_data_to_use
#         self.val_split_file_path = val_split_file_path
#         self.val_data_to_use = val_data_to_use
#         self.test_split_file_path = test_split_file_path
#         self.test_data_to_use = test_data_to_use
#         self.chunk_length_in_seconds = chunk_length_in_seconds
#         self.audio_file_suffix = audio_file_suffix
#         self.annotations_file_suffix = annotations_file_suffix
#         self.metadata_file_suffix = metadata_file_suffix
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory

#     def setup(self, stage: str) -> None:
#         if stage == "fit" or stage == "validate":
#             self.train_dataset = GreatestHitsDataset(
#                 root_dir=self.root_dir,
#                 split_file_path=self.train_split_file_path,
#                 split='train',
#                 data_to_use=self.train_data_to_use,
#                 chunk_length_in_seconds=self.chunk_length_in_seconds,
#                 audio_file_suffix=self.audio_file_suffix,
#                 annotations_file_suffix=self.annotations_file_suffix,
#                 metadata_file_suffix=self.metadata_file_suffix,
#             )

#             self.val_dataset = GreatestHitsDataset(
#                 root_dir=self.root_dir,
#                 split_file_path=self.val_split_file_path,
#                 split='val',
#                 data_to_use=self.val_data_to_use,
#                 chunk_length_in_seconds=self.chunk_length_in_seconds,
#                 audio_file_suffix=self.audio_file_suffix,
#                 annotations_file_suffix=self.annotations_file_suffix,
#                 metadata_file_suffix=self.metadata_file_suffix,
#             )

#             self.train_dataset.print()
#             self.val_dataset.print()

#         if stage == "test":
#             self.test_dataset = GreatestHitsDataset(
#                 root_dir=self.root_dir,
#                 split_file_path=self.test_split_file_path,
#                 split='test',
#                 data_to_use=self.test_data_to_use,
#                 chunk_length_in_seconds=self.chunk_length_in_seconds,
#                 audio_file_suffix=self.audio_file_suffix,
#                 annotations_file_suffix=self.annotations_file_suffix,
#                 metadata_file_suffix=self.metadata_file_suffix,
#             )

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             dataset=self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=True,
#             drop_last=True,
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             dataset=self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#             drop_last=False,
#         )

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             dataset=self.test_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#             drop_last=False,
#         )


# if __name__ == '__main__':
#     train_dataset = GreatestHitsDataset(
#         root="/import/c4dm-datasets-ext/GREATEST-HITS-DATASET/greatesthit-process-resized",
#         split_json_path="VideoOnset/data/greatesthit_train_2.00.json",
#         split='train',
#         repeat=1,
#         max_sample=0,
#         audio_file_suffix="mic_resampled.wav",
#     )

#     val_dataset = GreatestHitsDataset(
#         root="/import/c4dm-datasets-ext/GREATEST-HITS-DATASET/greatesthit-process-resized",
#         split_json_path="VideoOnset/data/greatesthit_val_2.00.json",
#         split='val',
#         repeat=1,
#         max_sample=0,
#         audio_file_suffix="mic_resampled.wav",
#     )

#     datamodule = Datamodule(
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         batch_size=32,
#         num_workers=8,
#         pin_memory=True,
#     )

#     model = Model(
#         lr=1e-4,
#         lr_beta1=0.95,
#         lr_beta2=0.999,
#         lr_eps=1e-6,
#         lr_weight_decay=1e-3,
#         model=VideoOnsetNet(
#             pretrained=False
#         ),
#     )

#     print(model)
#     print(f"num params: {sum(p.numel() for p in model.parameters())}")


####################################################################

# """ Callbacks """
# def get_wandb_logger(trainer: pl.Trainer) -> Optional[WandbLogger]:
#     """Safely get Weights&Biases logger from Trainer."""

#     if isinstance(trainer.logger, WandbLogger):
#         return trainer.logger

#     # if isinstance(trainer.logger, LoggerCollection):
#     #     for logger in trainer.logger:
#     #         if isinstance(logger, WandbLogger):
#     #             return logger

#     print("WandbLogger not found.")
#     return None


# def log_wandb_audio_batch(
#     logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
# ):
#     num_items = samples.shape[0]
#     samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
#     logger.log(
#         {
#             f"sample_{idx}_{id}": wandb.Audio(
#                 samples[idx],
#                 caption=caption,
#                 sample_rate=sampling_rate,
#             )
#             for idx in range(num_items)
#         }
#     )


# def log_wandb_audio_spectrogram(
#     logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
# ):
#     num_items = samples.shape[0]
#     samples = samples.detach().cpu()
#     transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=sampling_rate,
#         n_fft=1024,
#         hop_length=512,
#         n_mels=80,
#         center=True,
#         norm="slaney",
#     )

#     def get_spectrogram_image(x):
#         spectrogram = transform(x[0])
#         image = librosa.power_to_db(spectrogram)
#         trace = [go.Heatmap(z=image, colorscale="viridis")]
#         layout = go.Layout(
#             yaxis=dict(title="Mel Bin (Log Frequency)"),
#             xaxis=dict(title="Frame"),
#             title_text=caption,
#             title_font_size=10,
#         )
#         fig = go.Figure(data=trace, layout=layout)
#         return fig

#     logger.log(
#         {
#             f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
#             for idx in range(num_items)
#         }
#     )


# class SampleLogger(pl.Callback):
#     def __init__(
#         self,
#         # num_items: int,
#         # channels: int,
#         # sampling_rate: int,
#         # length: int
#     ) -> None:
#         # self.num_items = num_items
#         # self.channels = channels
#         # self.sampling_rate = sampling_rate
#         # self.length = length
#         self.log_next = False

#     def on_validation_epoch_start(self, trainer, pl_module):
#         self.log_next = True

#     def on_validation_batch_start(
#         self, trainer, pl_module, batch, batch_idx, dataloader_idx
#     ):
#         if self.log_next:
#             self.log_sample(trainer, pl_module, batch)
#             self.log_next = False

#     @torch.no_grad()
#     def log_sample(self, trainer, pl_module, batch):
#         is_train = pl_module.training
#         if is_train:
#             pl_module.eval()

#         # wandb_logger = get_wandb_logger(trainer).experiment

#         # log_wandb_audio_batch(
#         #     logger=wandb_logger,
#         #     id="sample",
#         #     samples=pred,
#         #     sampling_rate=self.sampling_rate,
#         #     caption=f"Sampled in {steps} steps",
#         # )
#         #
#         # log_wandb_audio_batch(
#         #     logger=wandb_logger,
#         #     id="cond",
#         #     samples=z,
#         #     sampling_rate=self.sampling_rate,
#         #     caption=f"Conditioning",
#         # )

#         if is_train:
#             pl_module.train()
