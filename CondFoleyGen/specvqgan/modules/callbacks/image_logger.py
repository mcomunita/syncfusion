import os
import random
import soundfile
import torchvision
import librosa
import torch
import numpy as np
import pytorch_lightning as pl

from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from CondFoleyGen.specvqgan.utils import instantiate_from_config


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        for_specs=False,
        vocoder_cfg=None,
        spec_dir_name=None,
        sample_rate=44100,
        epoch_frequency=1
    ):
        super().__init__()
        self.batch_freq = batch_frequency
        self.epoch_frequency = epoch_frequency
        self.max_images = max_images
        # self.logger_log_images = {
        #     pl.loggers.TestTubeLogger: self._testtube,
        # }
        self.logger_log_images = {
            pl.loggers.CSVLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.for_specs = for_specs
        self.spec_dir_name = spec_dir_name
        self.sample_rate = sample_rate
        print('We will not save audio for conditioning and conditioning_rec')
        if self.for_specs:
            self.vocoder = instantiate_from_config(vocoder_cfg)
            # self.vocoder = vocoder_cfg

    def _visualize_attention(self, attention, scale_by_prior=True):
        if scale_by_prior:
            B, H, T, T = attention.shape
            # attention weight is 1/T: if we have a seq with length 3 the weights are 1/3, 1/3, and 1/3
            # making T by T matrix with zeros in the upper triangular part
            attention_uniform_prior = 1 / torch.arange(1, T+1).view(1, T, 1).repeat(B, 1, T)
            attention_uniform_prior = attention_uniform_prior.tril().view(B, 1, T, T).to(attention.device)
            attention = attention - attention_uniform_prior

        attention_agg = attention.sum(dim=1, keepdims=True)
        return attention_agg

    def _log_rec_audio(self, specs, tag, global_step, pl_module=None, save_rec_path=None):

        # specs are (B, 1, F, T)
        for i, spec in enumerate(specs):
            spec = spec.data.squeeze(0).cpu().numpy()
            # audios are in [-1, 1], making them in [0, 1]
            spec = (spec + 1) / 2
            wave = self.vocoder.vocode(spec, global_step)
            wave = torch.from_numpy(wave).unsqueeze(0)
            if pl_module is not None:
                pl_module.logger.experiment.add_audio(f'{tag}_{i}', wave, pl_module.global_step, self.sample_rate)
            # in case we would like to save it on disk
            if save_rec_path is not None:
                try:
                    librosa.output.write_wav(save_rec_path, wave.squeeze(0).numpy(), self.sample_rate)
                except AttributeError:
                    soundfile.write(save_rec_path, wave.squeeze(0).numpy(), self.sample_rate, 'FLOAT')

    @rank_zero_only
    def _testtube(self, pl_module, images, batch, batch_idx, split):

        if pl_module.__class__.__name__ == 'Net2NetTransformer':
            cond_stage_model = pl_module.cond_stage_model.__class__.__name__
        else:
            cond_stage_model = None

        for k in images:
            tag = f'{split}/{k}'
            if cond_stage_model in ['ClassOnlyStage', 'FeatsClassStage'] and k in ['conditioning', 'conditioning_rec']:
                # saving the classes for the current batch
                pl_module.logger.experiment.add_text(tag, '; '.join(batch['label']))
                # breaking here because we don't want to call add_image
                if cond_stage_model == 'FeatsClassStage':
                    grid = torchvision.utils.make_grid(images[k]['feature'].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
                else:
                    continue
            elif k in ['att_nopix', 'att_half', 'att_det']:
                B, H, T, T = images[k].shape
                grid = torchvision.utils.make_grid(self._visualize_attention(images[k]), nrow=H, normalize=True)
            elif cond_stage_model in ['RawFeatsStage', 'VQModel1d', 'FeatClusterStage'] and k in ['conditioning', 'conditioning_rec']:
                grid = torchvision.utils.make_grid(images[k].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
            else:
                if self.for_specs:
                    # flipping values along frequency dim, otherwise mels are upside-down (1, F, T)
                    grid = torchvision.utils.make_grid(images[k].flip(dims=(2,)), nrow=1)
                    # also reconstruct waveform given the spec and inv_transform
                    if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
                        self._log_rec_audio(images[k], tag, pl_module.global_step, pl_module=pl_module)
                else:
                    grid = torchvision.utils.make_grid(images[k])
                # attention is already in [0, 1] therefore ignoring this line
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, pl_module, split, images, batch, batch_idx):
        root = os.path.join(pl_module.logger.save_dir, 'images', split)

        if pl_module.__class__.__name__ == 'Net2NetTransformer':
            cond_stage_model = pl_module.cond_stage_model.__class__.__name__
        else:
            cond_stage_model = None

        for k in images:
            if cond_stage_model in ['ClassOnlyStage', 'FeatsClassStage'] and k in ['conditioning', 'conditioning_rec']:
                filename = '{}_gs-{:06}_e-{:03}_b-{:06}.txt'.format(
                    k,
                    pl_module.global_step,
                    pl_module.current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                # saving the classes for the current batch
                with open(path, 'w') as file:
                    file.write('\n'.join(batch['label']))
                # next loop iteration here because we don't want to call add_image
                if cond_stage_model == 'FeatsClassStage':
                    grid = torchvision.utils.make_grid(images[k]['feature'].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
                else:
                    continue
            elif k in ['att_nopix', 'att_half', 'att_det']:  # GPT CLass
                B, H, T, T = images[k].shape
                grid = torchvision.utils.make_grid(self._visualize_attention(images[k]), nrow=H, normalize=True)
            elif cond_stage_model in ['RawFeatsStage', 'VQModel1d', 'FeatClusterStage'] and k in ['conditioning', 'conditioning_rec']:
                grid = torchvision.utils.make_grid(images[k].unsqueeze(1).permute(0, 1, 3, 2), nrow=1, normalize=True)
            else:
                if self.for_specs:
                    # flipping values along frequency dim, otherwise mels are upside-down (1, F, T)
                    grid = torchvision.utils.make_grid(images[k].flip(dims=(2,)), nrow=1)
                else:
                    grid = torchvision.utils.make_grid(images[k], nrow=4)
                # attention is already in [0, 1] therefore ignoring this line
                grid = (grid+1.0)/2.0  # -1,1 -> 0,1; c,h,w

            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = '{}_gs-{:06}_e-{:03}_b-{:06}.png'.format(
                k,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

            # also save audio on the disk
            if self.for_specs:
                tag = f'{split}/{k}'
                filename = filename.replace('.png', '.wav')
                path = os.path.join(root, filename)
                if k not in ['conditioning', 'conditioning_rec', 'att_nopix', 'att_half', 'att_det']:
                    self._log_rec_audio(images[k], tag, pl_module.global_step, save_rec_path=path)

    def log_img(self, pl_module, batch, batch_idx, split='train'):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, 'log_images') and
                callable(pl_module.log_images) and
                self.max_images > 0 and
                pl_module.first_stage_key != 'feature' and
                pl_module.current_epoch % self.epoch_frequency == 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                if isinstance(images[k], dict):
                    N = min(images[k]['feature'].shape[0], self.max_images)
                    images[k]['feature'] = images[k]['feature'][:N]
                    if isinstance(images[k]['feature'], torch.Tensor):
                        images[k]['feature'] = images[k]['feature'].detach().cpu()
                        if self.clamp:
                            images[k]['feature'] = torch.clamp(images[k]['feature'], -1., 1.)
                else:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module, split, images, batch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, batch, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img(pl_module, batch, batch_idx, split='val')
