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
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def common_step(self, batch, batch_idx, mode="train"):
        frames, labels = batch['frames'], batch['label']
        pred = self.model(frames)
        loss = self.loss(pred, labels)
        metrics = self.loss.evaluate(pred, labels)
        return loss, metrics
        
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
        self.log_metrics(metrics, mode="train")

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
        """merge content of all files with the same video name
        and save to a single file
        """
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
        """Computes the average precision, binary accuracy 
        and onset number accuracy of the model.
        """
        ons_num_acc = self.onset_num_acc(pred, target)

        # flatten
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        pred = torch.sigmoid(pred)

        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()

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
        
        return res

    def binary_acc(self, pred, target):
        pred[pred > self.threshold] = 1
        pred[pred <= self.threshold] = 0
        acc = np.sum(pred == target) / target.shape[0]
        return acc

    def onset_num_acc(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.data.cpu().numpy()
        target = target.data.cpu().numpy()

        pred[pred > self.threshold] = 1
        pred[pred <= self.threshold] = 0

        pred = pred.astype(int)
        target = target.astype(int)

        # remove consecutive onsets in pred
        for i in range(pred.shape[0]):
            for j in range(pred.shape[-1] - 1):
                if pred[i][j] == 1 and pred[i][j + 1] == 1:
                    pred[i][j + 1] = 0

        pred = np.sum(pred, axis=-1)
        target = np.sum(target, axis=-1)

        acc_num_onset = np.sum(pred == target)
        
        return acc_num_onset / len(pred)