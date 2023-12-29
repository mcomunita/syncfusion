import os
import soundfile
import numpy as np
import torch
import time
import torchvision.transforms as transforms
from pprint import pprint


from PIL import Image
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import default_collate
from audio_generation_utils import (CropImage,
                                    load_specs_as_img,
                                    attach_audio_to_video,
                                    draw_spec,
                                    run_style_transfer)
from sample_visualization import spec_to_audio_to_streamlit
from feature_extraction.demo_utils import (extract_melspectrogram,
                                           trim_video,
                                           load_frames,
                                           reencode_video_with_diff_fps)
from specvqgan.data.transforms import (Resize3D,
                                       CenterCrop3D,
                                       ToTensor3D,
                                       Normalize3D)

# ----------------------
# Constants
# ----------------------
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

FRAME_TRANS = transforms.Compose([
    Resize3D(128),
    CenterCrop3D(112),
    ToTensor3D(),
    Normalize3D(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ------------------------------
# Single Output (no re-ranking)
# ------------------------------


def generate_audio_using_ref_video_and_cond_audiovideo(
    config,
    sampler_model,
    vocoder_model,
    dataset_item,
    target_log_dir,
    temp_dir,
    original_video_dir,
    original_video_suffix,
    sample_rate,
    frame_rate,
    chunk_length_in_seconds,
    vqgan_codebook_length_in_seconds=2.0,
    normalize=False,
    using_torch=False,
    show_griffin_lim=False,
    n_spectrogram_time_frames=160,
    # denoise_audio=False,
    W_scale=1,
    slide_win_mode='half',
    fix_cond_length=True,
    temperature=1.0,
    device='cpu'
):
    print("datatset_item")
    print(f"feats_: {dataset_item['file_path_feats_']}")
    print(f"cond_feats_: {dataset_item['file_path_cond_feats_']}")
    print(f"audio_: {dataset_item['file_path_wav_']}")
    print(f"cond_audio_: {dataset_item['file_path_cond_wav_']}")

    # set in/out dimensions
    # L = int(chunk_length_in_seconds * W_scale)
    # vqgan_L = int(vqgan_codebook_length_in_seconds * W_scale)
    L = int(chunk_length_in_seconds)
    vqgan_L = int(vqgan_codebook_length_in_seconds)
    n_spectrogram_time_frames = int(n_spectrogram_time_frames * W_scale)

    print("L: ", L)
    print("vqgan_L: ", vqgan_L)
    print("n_spectrogram_time_frames: ", n_spectrogram_time_frames)
    print("W_scale: ", W_scale)

    # extracts audio and frames from item
    ref_and_cond_frames = dataset_item["feature"]
    ref_audio = dataset_item["image"]
    cond_audio = dataset_item["cond_image"]

    # --- get REF and COND VIDEO EMBEDDINGS using video encoder
    frames = ref_and_cond_frames.unsqueeze(0)  # r2plus1d needs (N, 3, T, 112, 112)
    ref_and_cond_frames = {"feature": frames.to(device)}  # encoder expects dict

    with torch.no_grad():
        c = sampler_model.get_input(
            key=sampler_model.cond_stage_key,
            batch=ref_and_cond_frames
        )

    # --- get REF AUDIO TOKENS
    # get spectrogram
    spectrogram = extract_melspectrogram(
        ref_audio,
        sample_rate,
        normalize=normalize,
        using_torch=using_torch,
        duration=vqgan_L,
    )

    # format spectrogram
    spec_H, spec_W = spectrogram.shape
    if spec_W > n_spectrogram_time_frames:  # limit number of frequency bins in spectrogram
        spectrogram = spectrogram[:, :n_spectrogram_time_frames]
    else:  # or pad spectrogram
        pad = np.zeros((spec_H, n_spectrogram_time_frames), dtype=spectrogram.dtype)
        pad[:, :spec_W] = spectrogram
        spectrogram = pad

    if config["model"]["params"]["spec_crop_len"] is None or W_scale != 1:
        config["model"]["params"]["spec_crop_len"] = n_spectrogram_time_frames
    if spectrogram.shape[1] > config["model"]["params"]["spec_crop_len"]:
        crop_img_fn = CropImage([
            config["model"]["params"]["mel_num"], config["model"]["params"]["spec_crop_len"]
        ])
        spectrogram = {'input': spectrogram}  # CropImage expects dict
        spectrogram = crop_img_fn(spectrogram)
        spectrogram = spectrogram['input']  # CropImage returns dict

    # prepare input
    spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # encoder expects (N, 1, 80, 160)
    batch = {"image": spectrogram.to(device)}  # encoder expects dict

    # get embedding from input spectrogram
    with torch.no_grad():
        x = sampler_model.get_input(
            key=sampler_model.first_stage_key,
            batch=batch
        )
        # mel_x = x.detach().cpu().numpy()

    # encode and decode the spectrogram
    with torch.no_grad():
        quant_z, z_indices = sampler_model.encode_to_z(x)
        xrec = sampler_model.first_stage_model.decode(quant_z)
        mel_xrec = xrec.detach().cpu().numpy()

    # --- get COND AUDIO TOKENS
    # get spectrogram
    spectrogram = extract_melspectrogram(
        cond_audio,
        sample_rate,
        normalize=normalize,
        using_torch=using_torch,
        duration=vqgan_L,
    )

    # format spectrogram
    spec_H, spec_W = spectrogram.shape
    if spec_W > n_spectrogram_time_frames:  # limit number of frequency bins in spectrogram
        padded = False
        spectrogram = spectrogram[:, :n_spectrogram_time_frames]
    else:  # or pad spectrogram
        padded = True
        pad = np.zeros((spec_H, n_spectrogram_time_frames), dtype=spectrogram.dtype)
        pad[:, :spec_W] = spectrogram
        orig_width = spec_W
        spectrogram = pad

    if config["model"]["params"]["spec_crop_len"] is None or W_scale != 1:
        config["model"]["params"]["spec_crop_len"] = n_spectrogram_time_frames
    if spectrogram.shape[1] > config["model"]["params"]["spec_crop_len"]:
        crop_img_fn = CropImage([
            config["model"]["params"]["mel_num"], config["model"]["params"]["spec_crop_len"]
        ])
        spectrogram = {'input': spectrogram}  # CropImage expects dict
        spectrogram = crop_img_fn(spectrogram)
        spectrogram = spectrogram['input']  # CropImage returns dict

    # prepare input
    spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)  # encoder expects (N, 1, 80, 160)
    batch = {"cond_image": spectrogram.to(device)}  # encoder expects dict

    # get embedding from cond spectrogram
    with torch.no_grad():
        xp = sampler_model.get_input(
            key=sampler_model.cond_first_stage_key,
            batch=batch
        )
        mel_xp = xp.detach().cpu().numpy()

    # encode and decode the spectrogram
    with torch.no_grad():
        quant_zp, zp_indices = sampler_model.encode_to_z(xp)
        xprec = sampler_model.first_stage_model.decode(quant_zp)
        mel_xprec = xprec.detach().cpu().numpy()

    # --- SAMPLING
    # define sampling parameters
    top_x = sampler_model.first_stage_model.quantize.n_e // 2  # take codes

    # start sampling
    with torch.no_grad():
        start_t = time.time()

        quant_c, c_indices = sampler_model.encode_to_c(c)
        z_indices_clip = z_indices[:, :sampler_model.clip * W_scale]
        zp_indices_clip = zp_indices[:, :sampler_model.clip * W_scale]
        z_indices_rec = z_indices.clone()
        # crec = sampler.cond_stage_model.decode(quant_c)

        patch_size_i = 5
        c_window_size = int(2 * frame_rate)

        downsampled_size = n_spectrogram_time_frames // 16
        cond_patch_shift_j = (W_scale - 1) * (downsampled_size // W_scale)

        if 'dropcond_' in target_log_dir:
            B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(downsampled_size))
            patch_size_j = int(downsampled_size // W_scale)
        else:
            B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(2*downsampled_size))
            patch_size_j = int(2*downsampled_size // W_scale)

        z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(device)

        if 'dropcond_' not in target_log_dir:
            start_step = zp_indices_clip.shape[1]
            z_pred_indices[:, :start_step] = zp_indices_clip[:, :start_step]
        elif 'dropcond_' in target_log_dir:
            start_step = 0

        pbar = tqdm(range(start_step, hr_w * hr_h), desc='Sampling Codebook Indices')
        for step in pbar:
            i = step % hr_h
            j = step // hr_h

            i_start = min(max(0, i - (patch_size_i // 2)), hr_h - patch_size_i)
            # only last
            #
            if slide_win_mode == 'half':
                j_start = min(max(0, j - (3 * patch_size_j // 4)), hr_w - patch_size_j)
            elif slide_win_mode == 'last':
                j_start = min(max(0, j - patch_size_j + 1), hr_w - patch_size_j)
            else:
                raise NotImplementedError
            i_end = i_start + patch_size_i
            j_end = j_start + patch_size_j

            local_i = i - i_start

            patch_2d_shape = (B, D, patch_size_i, patch_size_j)

            if W_scale != 1:
                cond_j_start = 0 if fix_cond_length else max(0, j_start - cond_patch_shift_j)
                cond_j_end = cond_j_start + (patch_size_j // 2)
                tar_j_start = max(0, j_start - cond_patch_shift_j) + (hr_w // 2)
                tar_j_end = tar_j_start + (patch_size_j // 2)

                local_j = j - tar_j_start + (patch_size_j // 2)

                pbar.set_postfix(
                    Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end}, {cond_j_start}:{cond_j_end}|{tar_j_start}:{tar_j_end})'
                )
                cond_patch = z_pred_indices \
                    .reshape(B, hr_w, hr_h) \
                    .permute(0, 2, 1)[:, i_start:i_end, cond_j_start:cond_j_end].permute(0, 2, 1)
                tar_patch = z_pred_indices \
                    .reshape(B, hr_w, hr_h) \
                    .permute(0, 2, 1)[:, i_start:i_end, tar_j_start:tar_j_end].permute(0, 2, 1)
                patch = torch.cat([cond_patch, tar_patch], dim=1).reshape(B, patch_size_i * patch_size_j)

                cond_t_start = cond_j_start * 0.2
                cond_frame_start = int(cond_t_start * frame_rate)
                cond_frame_end = cond_frame_start + c_window_size
                tar_frame_start = int(cond_frame_start + c_window_size * W_scale)
                tar_frame_end = tar_frame_start + c_window_size
                cpatch = torch.cat([c_indices[:, :, cond_frame_start:cond_frame_end], c_indices[:, :, tar_frame_start:tar_frame_end]], dim=2)
            else:
                local_j = j - j_start
                pbar.set_postfix(
                    Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})'
                )
                patch = z_pred_indices
                # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
                cpatch = c_indices

            logits, _, attention = sampler_model.transformer(patch[:, :-1], cpatch)

            # remove conditioning
            logits = logits[:, -patch_size_j*patch_size_i:, :]

            local_pos_in_flat = local_j * patch_size_i + local_i
            logits = logits[:, local_pos_in_flat, :]

            logits = logits / temperature
            logits = sampler_model.top_k_logits(logits, top_x)

            # apply softmax to convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # sample from the distribution
            ix = torch.multinomial(probs, num_samples=1)
            z_pred_indices[:, j * hr_h + i] = ix

        # quant_z_shape = sampling_shape
        if 'dropcond_' in target_log_dir:
            z_indices_rec[:, :sampler_model.clip * W_scale] = z_pred_indices
        else:
            z_indices_rec[:, :sampler_model.clip * W_scale] = z_pred_indices[:, sampler_model.clip * W_scale:]
        # print(z_indices_rec)
        z_pred_img = sampler_model.decode_to_img(
            z_indices_rec,
            (1, 256, 5, 53 if vqgan_L == 10.0 else downsampled_size)
        )
        mel_z = z_pred_img.detach().cpu().numpy()

        with torch.no_grad():
            config["data"]["init_args"]["spec_dir_path"] = 'melspec_10s_22050hz'

            if padded:
                z_pred_img = z_pred_img[:, :, :, :orig_width]
                xrec = xrec[:, :, :, :orig_width]
                xprec = xprec[:, :, :, :orig_width]
                n_spectrogram_time_frames = orig_width

            waves = spec_to_audio_to_streamlit(
                z_pred_img,
                config["data"]["init_args"]["spec_dir_path"],
                sample_rate,
                show_griffin_lim=False,
                vocoder=vocoder_model,
                show_in_streamlit=False
            )

            # original reconstruction
            orig_waves = spec_to_audio_to_streamlit(
                xrec,
                config["data"]["init_args"]["spec_dir_path"],
                sample_rate,
                show_griffin_lim=False,
                vocoder=vocoder_model,
                show_in_streamlit=False
            )

            # conditional reconstruction
            cond_waves = spec_to_audio_to_streamlit(
                xprec,
                config["data"]["init_args"]["spec_dir_path"],
                sample_rate,
                show_griffin_lim=False,
                vocoder=vocoder_model,
                show_in_streamlit=False
            )
        
        # --- SAVE OUTPUTS
    
        print("waves: ", waves['vocoder'].shape)
        print("orig_waves: ", orig_waves['vocoder'].shape)
        print("cond_waves: ", cond_waves['vocoder'].shape) 

        waves['vocoder'] = waves['vocoder'][:int(sample_rate * L)]
        orig_waves['vocoder'] = orig_waves['vocoder'][:int(sample_rate * L)]
        cond_waves['vocoder'] = cond_waves['vocoder'][:int(sample_rate * L)]

        print("waves: ", waves['vocoder'].shape)
        print("orig_waves: ", orig_waves['vocoder'].shape)
        print("cond_waves: ", cond_waves['vocoder'].shape)

        # --- SAVE OUTPUTS
        # prepare output dirs
        os.makedirs(f'target_log_dir', exist_ok=True)

        target_gen_audio_dir = os.path.join(target_log_dir, f"generated_audio")
        target_gen_video_dir = os.path.join(target_log_dir, f"generated_video")
        target_cond_audio_dir = os.path.join(target_log_dir, f"cond_audio")
        target_cond_video_dir = os.path.join(target_log_dir, f"cond_video")
        target_orig_audio_dir = os.path.join(target_log_dir, f"orig_audio")
        target_orig_video_dir = os.path.join(target_log_dir, f"orig_video")

        os.makedirs(target_gen_audio_dir, exist_ok=True)
        os.makedirs(target_gen_video_dir, exist_ok=True)
        os.makedirs(target_cond_audio_dir, exist_ok=True)
        os.makedirs(target_cond_video_dir, exist_ok=True)
        os.makedirs(target_orig_audio_dir, exist_ok=True)
        os.makedirs(target_orig_video_dir, exist_ok=True)

        ref_video_name = dataset_item['file_path_feats_'][0].split('/')[-2]
        ref_video_start_frame = dataset_item['file_path_feats_'][1]

        cond_video_name = dataset_item['file_path_cond_feats_'][0].split('/')[-2]
        cond_video_start_frame = dataset_item['file_path_cond_feats_'][1]

        # ref_video_name = ref_video_path.split('/')[-2]
        # cond_video_name = cond_video_path.split('/')[-2]

        # source videos paths
        orig_video_path = os.path.join(original_video_dir, f"{ref_video_name}{original_video_suffix}")
        cond_video_path = os.path.join(original_video_dir, f"{cond_video_name}{original_video_suffix}")

        # save generated audio
        save_gen_audio_path = os.path.join(target_gen_audio_dir, f"{ref_video_name}_to_{cond_video_name}.wav")
        soundfile.write(save_gen_audio_path, waves['vocoder'], sample_rate, 'PCM_24')
        print(f'\nGenerated audio saved @ {save_gen_audio_path}')

        # save original video w/ generated audio
        save_gen_video_path = os.path.join(target_gen_video_dir, f"{ref_video_name}_to_{cond_video_name}.mp4")
        attach_audio_to_video(
            video_path=orig_video_path,
            audio_path=save_gen_audio_path,
            dest=save_gen_video_path,
            FPS=frame_rate,
            video_start_in_seconds=ref_video_start_frame / frame_rate,
            video_duration_in_seconds=L
        )
        print(f'Generated video saved @ {save_gen_video_path}')

        # save original audio
        save_orig_audio_path = os.path.join(target_orig_audio_dir, f"{ref_video_name}.wav")
        soundfile.write(save_orig_audio_path, orig_waves['vocoder'], sample_rate, 'PCM_24')
        print(f'Original audio saved @ {save_orig_audio_path}')

        # save original video w/ original audio
        save_orig_video_path = os.path.join(target_orig_video_dir, f"{ref_video_name}.mp4")
        attach_audio_to_video(
            video_path=orig_video_path,
            audio_path=save_orig_audio_path,
            dest=save_orig_video_path,
            FPS=frame_rate,
            video_start_in_seconds=ref_video_start_frame / frame_rate,
            video_duration_in_seconds=L
        )
        print(f'Original video saved @ {save_orig_video_path}')

        # save conditioning audio
        save_cond_audio_path = os.path.join(target_cond_audio_dir, f"{cond_video_name}.wav")
        soundfile.write(save_cond_audio_path, cond_waves['vocoder'], sample_rate, 'PCM_24')
        print(f'Conditioning audio saved @ {save_orig_audio_path}')

        # save conditioning video w/ conditioning audio
        save_cond_video_path = os.path.join(target_cond_video_dir, f"{cond_video_name}.mp4")
        attach_audio_to_video(
            video_path=cond_video_path,
            audio_path=save_cond_audio_path,
            dest=save_cond_video_path,
            FPS=frame_rate,
            video_start_in_seconds=cond_video_start_frame / frame_rate,
            video_duration_in_seconds=L
        )
        print(f'Conditioning video saved @ {save_orig_video_path}')

        # plot melspec
        # original
        plt.imshow(mel_xrec[0, 0, :, :n_spectrogram_time_frames], cmap='coolwarm', origin='lower')
        plt.axis('off')
        plt.savefig(save_orig_video_path.replace('.mp4', '.jpg'), bbox_inches='tight', pad_inches=0.)
        plt.close()

        # condition
        _spec_take_first = int(n_spectrogram_time_frames // W_scale) if fix_cond_length else n_spectrogram_time_frames
        plt.imshow(mel_xprec[0, 0, :, :_spec_take_first], cmap='coolwarm', origin='lower')
        plt.axis('off')
        plt.savefig(save_cond_video_path.replace('.mp4', '.jpg'), bbox_inches='tight', pad_inches=0.)
        plt.close()

        # generated
        draw_spec(mel_z[0, 0, :, :n_spectrogram_time_frames], save_gen_video_path.replace('.mp4', '.jpg'), cmap='coolwarm')

        return


# ------------------------------
# Single Output (no re-ranking)
# ------------------------------
def gen_audio_condImage_fast(
    ref_video_path,
    cond_video_path,
    model,
    device,
    target_log_dir='CondAVTransformer',
    cond_cnt=0,
    SR=22050,
    FPS=15,
    L=2.0,
    normalize=False,
    using_torch=False,
    show_griffin_lim=False,
    vqgan_L=10.0,
    style_transfer=False,
    target_start_time=0,
    cond_start_time=0,
    outside=False,
    remove_noise=False,
    spec_take_first=160,
    W_scale=1,
    slide_win_mode='half',
    temperature=1.0,
    ignore_input_spec=False,
    tmp_path='./tmp',
    fix_cond=True,
):
    '''
    parameters:
        ref_video_path: path to the target video, will be trimmed to 2s and re-encode into 15 fps.
        cond_video_path: path to the conditional video, will be trimmed to 2s and re-encode into 15 fps.
        model: model object, returned by load_model function
        target_log_dir: target output dir name in the 'logs' directory, e.g. output will be saved to 'logs/<target_log_dir>'
        cond_cnt: index of current condition video
        SR: sampling rate
        FPS: Frame rate
        L: length of generated sound
        normalize: whether to normaliza input waveform
        using_torch: use torchaudio to extrac spectrogram
        show_griffin_lim: use griffin_lim algorithm vocoder
        vqgan_L: length of VQ-GAN codebook, use 2 if using GreatestHit codebook
        style_transfer: generate style transfer sound
        target_start_time: if target video is from outside, trim from <target_start_time> to <target_start_time>+2
        cond_start_time: if conditional video is from outside, trim from <cond_start_time> to <cond_start_time>+2
        outside: indicate whether the video from outside source
        remove_noise: denoise for outside videos
        spec_take_first: size of the spectrogram to use
        W_scale: scale of audio duration as multiples of 2sec
        slide_win_mode: mode of sliding window, choose from ['half', 'last']
        temperature: temperature of multinomial sampling.
        ignore_input_spec: ignore input spec when input video is silent
        tmp_path: tmp dir to save intermediate files
        fix_cond: use only 2 sec condition regardless to input length.
    '''

    # config, sampler, melgan, melception = model
    config, sampler, melgan = model

    # Set in/out lengths
    L = int(L * W_scale)
    vqgan_L = int(vqgan_L * W_scale)
    spec_take_first = int(spec_take_first * W_scale)

    # Load reference video frames (first FPS * L frames)
    if '_denoised_' not in ref_video_path or outside:
        new_fps_video_path = reencode_video_with_diff_fps(ref_video_path, tmp_path, FPS)
        new_ref_video_path = trim_video(new_fps_video_path, target_start_time, vqgan_L, tmp_path=tmp_path)
        ref_video_path = new_ref_video_path
    frames = [Image.fromarray(f) for f in load_frames(ref_video_path)][:int(FPS * L)]
    frames = FRAME_TRANS(frames)

    # Load conditioning video frames (first FPS * L frames)
    if '_denoised_' not in cond_video_path or outside:
        new_fps_video_path = reencode_video_with_diff_fps(cond_video_path, tmp_path, FPS)
        new_cond_video_path = trim_video(new_fps_video_path, cond_start_time, vqgan_L, tmp_path=tmp_path)
        cond_video_path = new_cond_video_path
    cond_frames = [Image.fromarray(f) for f in load_frames(cond_video_path)][:int(FPS * L)]
    cond_frames = FRAME_TRANS(cond_frames)

    # --- Prepare Batch
    visual_features = {'feature': np.stack(cond_frames + frames, axis=0)}
    batch = default_collate([visual_features])
    batch['feature'] = batch['feature'].to(device)
    with torch.no_grad():
        c = sampler.get_input(sampler.cond_stage_key, batch)

    # --- Reference Input
    # extract spectrogram
    if not ignore_input_spec:
        spectrogram = extract_melspectrogram(
            ref_video_path,
            SR,
            normalize=normalize,
            using_torch=using_torch,
            remove_noise=remove_noise,
            duration=vqgan_L,
            tmp_path=tmp_path
        )

        # format spectrogram
        spec_H, spec_W = spectrogram.shape
        if spec_W > spec_take_first:  # limit number of frequency bins in spectrogram
            spectrogram = spectrogram[:, :spec_take_first]
        else:  # or pad spectrogram
            pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
            pad[:, :spec_W] = spectrogram
            spectrogram = pad

        spectrogram = {'input': spectrogram}

        if config.data.params.spec_crop_len is None or W_scale != 1:
            config.data.params.spec_crop_len = spec_take_first
        if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
            random_crop = False
            crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
            spectrogram = crop_img_fn(spectrogram)

        # prepare input
        batch = default_collate([spectrogram])
        batch['image'] = batch['input'].to(device)

        # get tokens from input spectrogram
        x = sampler.get_input(sampler.first_stage_key, batch)
        mel_x = x.detach().cpu().numpy()

        # encode and decode the spectrogram
        with torch.no_grad():
            quant_z, z_indices = sampler.encode_to_z(x)
            # print(z_indices)
            xrec = sampler.first_stage_model.decode(quant_z)
            mel_xrec = xrec.detach().cpu().numpy()

    # --- Conditioning
    # extract spectrogram
    spectrogram = extract_melspectrogram(
        cond_video_path,
        SR,
        normalize=normalize,
        using_torch=using_torch,
        remove_noise=remove_noise,
        duration=vqgan_L,
        tmp_path=tmp_path
    )

    # format spectrogram
    spec_H, spec_W = spectrogram.shape
    if spec_W > spec_take_first:
        padded = False
        spectrogram = spectrogram[:, :spec_take_first]
    else:
        padded = True
        pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
        pad[:, :spec_W] = spectrogram
        orig_width = spec_W
        spectrogram = pad

    spectrogram = {'input': spectrogram}

    if config.data.params.spec_crop_len is None or W_scale != 1:
        config.data.params.spec_crop_len = spec_take_first
    if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
        random_crop = False
        crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
        spectrogram = crop_img_fn(spectrogram)

    # prepare input
    batch = default_collate([spectrogram])
    batch['cond_image'] = batch['input'].to(device)
    xp = sampler.get_input(sampler.cond_first_stage_key, batch)
    mel_xp = xp.detach().cpu().numpy()

    # encode and decode the Spectrogram
    with torch.no_grad():
        quant_zp, zp_indices = sampler.encode_to_z(xp)
        # print(zp_indices)
        xprec = sampler.first_stage_model.decode(quant_zp)
        mel_xprec = xprec.detach().cpu().numpy()

    if ignore_input_spec:
        z_indices = torch.zeros_like(zp_indices)
        xrec = torch.zeros_like(xprec)
        mel_xrec = np.zeros_like(mel_xprec)

    # --- Define Sampling Parameters
    # take top 1024 / 512 code
    top_x = sampler.first_stage_model.quantize.n_e // 2

    # --- Prepare Output Directory
    # if not os.path.exists(f'logs/{target_log_dir}'):
    os.makedirs(f'target_log_dir', exist_ok=True)

    target_dir = os.path.join(target_log_dir, f"2sec_full_generated_sound_{cond_cnt}")
    target_v_dir = os.path.join(target_log_dir, f"2sec_full_generated_video_{cond_cnt}")
    target_cond_v_dir = os.path.join(target_log_dir, f"2sec_full_cond_video_{cond_cnt}")
    target_orig_v_dir = os.path.join(target_log_dir, f"2sec_full_orig_video")

    # if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    # if not os.path.exists(target_v_dir):
    os.makedirs(target_v_dir, exist_ok=True)
    # if not os.path.exists(target_cond_v_dir):
    os.makedirs(target_cond_v_dir, exist_ok=True)
    # if not os.path.exists(target_orig_v_dir):
    os.makedirs(target_orig_v_dir, exist_ok=True)

    # --- Start Sampling
    if style_transfer:
        content_img = load_specs_as_img(mel_xrec[0, 0, :, :spec_take_first])
        style_img = load_specs_as_img(mel_xprec[0, 0, :, :spec_take_first])
        generated_spec = run_style_transfer(
            cnn_normalization_mean.to(),
            cnn_normalization_std.to(),
            content_img.clone().to(device),
            style_img.clone().to(device),
            content_img.clone().to(device),
        )
        z_pred_img = torch.mean(generated_spec, dim=1, keepdim=True)
        mel_z = z_pred_img.detach().cpu().numpy()
    else:
        with torch.no_grad():
            start_t = time.time()

            quant_c, c_indices = sampler.encode_to_c(c)
            z_indices_clip = z_indices[:, :sampler.clip * W_scale]
            zp_indices_clip = zp_indices[:, :sampler.clip * W_scale]
            z_indices_rec = z_indices.clone()
            # crec = sampler.cond_stage_model.decode(quant_c)

            patch_size_i = 5
            c_window_size = int(2 * FPS)

            downsampled_size = spec_take_first // 16
            cond_patch_shift_j = (W_scale - 1) * (downsampled_size // W_scale)
            if 'dropcond_' in target_log_dir:
                B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(downsampled_size))
                patch_size_j = int(downsampled_size // W_scale)
            else:
                B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(2*downsampled_size))
                patch_size_j = int(2*downsampled_size // W_scale)
            z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(device)

            if 'dropcond_' not in target_log_dir:
                start_step = zp_indices_clip.shape[1]
                z_pred_indices[:, :start_step] = zp_indices_clip[:, :start_step]
            elif 'dropcond_' in target_log_dir:
                start_step = 0

            pbar = tqdm(range(start_step, hr_w * hr_h), desc='Sampling Codebook Indices')
            for step in pbar:
                i = step % hr_h
                j = step // hr_h

                i_start = min(max(0, i - (patch_size_i // 2)), hr_h - patch_size_i)
                # only last
                #
                if slide_win_mode == 'half':
                    j_start = min(max(0, j - (3 * patch_size_j // 4)), hr_w - patch_size_j)
                elif slide_win_mode == 'last':
                    j_start = min(max(0, j - patch_size_j + 1), hr_w - patch_size_j)
                else:
                    raise NotImplementedError
                i_end = i_start + patch_size_i
                j_end = j_start + patch_size_j

                local_i = i - i_start

                patch_2d_shape = (B, D, patch_size_i, patch_size_j)

                if W_scale != 1:
                    cond_j_start = 0 if fix_cond else max(0, j_start - cond_patch_shift_j)
                    cond_j_end = cond_j_start + (patch_size_j // 2)
                    tar_j_start = max(0, j_start - cond_patch_shift_j) + (hr_w // 2)
                    tar_j_end = tar_j_start + (patch_size_j // 2)

                    local_j = j - tar_j_start + (patch_size_j // 2)

                    pbar.set_postfix(
                        Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end}, {cond_j_start}:{cond_j_end}|{tar_j_start}:{tar_j_end})'
                    )
                    cond_patch = z_pred_indices \
                        .reshape(B, hr_w, hr_h) \
                        .permute(0, 2, 1)[:, i_start:i_end, cond_j_start:cond_j_end].permute(0, 2, 1)
                    tar_patch = z_pred_indices \
                        .reshape(B, hr_w, hr_h) \
                        .permute(0, 2, 1)[:, i_start:i_end, tar_j_start:tar_j_end].permute(0, 2, 1)
                    patch = torch.cat([cond_patch, tar_patch], dim=1).reshape(B, patch_size_i * patch_size_j)

                    cond_t_start = cond_j_start * 0.2
                    cond_frame_start = int(cond_t_start * FPS)
                    cond_frame_end = cond_frame_start + c_window_size
                    tar_frame_start = int(cond_frame_start + c_window_size * W_scale)
                    tar_frame_end = tar_frame_start + c_window_size
                    cpatch = torch.cat([c_indices[:, :, cond_frame_start:cond_frame_end], c_indices[:, :, tar_frame_start:tar_frame_end]], dim=2)
                else:
                    local_j = j - j_start
                    pbar.set_postfix(
                        Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})'
                    )
                    patch = z_pred_indices
                    # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
                    cpatch = c_indices

                logits, _, attention = sampler.transformer(patch[:, :-1], cpatch)
                # remove conditioning
                logits = logits[:, -patch_size_j*patch_size_i:, :]

                local_pos_in_flat = local_j * patch_size_i + local_i
                logits = logits[:, local_pos_in_flat, :]

                logits = logits / temperature
                logits = sampler.top_k_logits(logits, top_x)

                # apply softmax to convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # sample from the distribution
                ix = torch.multinomial(probs, num_samples=1)
                z_pred_indices[:, j * hr_h + i] = ix

            # quant_z_shape = sampling_shape
            if 'dropcond_' in target_log_dir:
                z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices
            else:
                z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices[:, sampler.clip * W_scale:]
            # print(z_indices_rec)
            z_pred_img = sampler.decode_to_img(z_indices_rec,
                                               (1, 256, 5, 53 if vqgan_L == 10.0 else downsampled_size))
            mel_z = z_pred_img.detach().cpu().numpy()

    with torch.no_grad():
        config.data.params.spec_dir_path = 'melspec_10s_22050hz'

        if padded:
            z_pred_img = z_pred_img[:, :, :, :orig_width]
            xrec = xrec[:, :, :, :orig_width]
            xprec = xprec[:, :, :, :orig_width]
            spec_take_first = orig_width

        waves = spec_to_audio_to_streamlit(
            z_pred_img,
            config.data.params.spec_dir_path,
            config.data.params.sample_rate,
            show_griffin_lim=show_griffin_lim,
            vocoder=melgan,
            show_in_st=False
        )

        # Original Reconstruction
        orig_waves = spec_to_audio_to_streamlit(
            xrec,
            config.data.params.spec_dir_path,
            config.data.params.sample_rate,
            show_griffin_lim=show_griffin_lim,
            vocoder=melgan,
            show_in_st=False
        )

        # Conditional Reconstruction
        cond_waves = spec_to_audio_to_streamlit(
            xprec,
            config.data.params.spec_dir_path,
            config.data.params.sample_rate,
            show_griffin_lim=show_griffin_lim,
            vocoder=melgan,
            show_in_st=False
        )

    if show_griffin_lim:
        waves['vocoder'] = waves['inv_transforms'][:int(22050 * L)]
    else:
        waves['vocoder'] = waves['vocoder'][:int(22050 * L)]

    _cond_video_path = cond_video_path
    save_path = os.path.join(target_dir, Path(ref_video_path).stem +
                             '_to_' + Path(_cond_video_path).stem + '.wav')

    soundfile.write(save_path, waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
    print(f'The sample has been saved @ {save_path}')

    save_video_path = os.path.join(target_v_dir, Path(ref_video_path).stem +
                                   '_to_' + Path(_cond_video_path).stem + '.mp4')
    attach_audio_to_video(ref_video_path, save_path, save_video_path, 0, v_duration=L)

    # Original sound attach
    if show_griffin_lim:
        waves['vocoder'] = waves['inv_transforms'][:int(22050 * L)]
    else:
        waves['vocoder'] = waves['vocoder'][:int(22050 * L)]
    orig_save_path = os.path.join(target_orig_v_dir, Path(ref_video_path).stem + '.wav')
    soundfile.write(orig_save_path, orig_waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
    print(f'The sample has been saved @ {orig_save_path}')
    orig_save_video_path = os.path.join(target_orig_v_dir, Path(ref_video_path).stem + '.mp4')
    attach_audio_to_video(ref_video_path, orig_save_path, orig_save_video_path, 0, recon_only=True, v_duration=L)

    # Conditional sound attach
    _cond_video_path = cond_video_path
    # Save only the first 2sec conditional audio+video if fix_cond
    _L = L // W_scale if fix_cond else L
    if show_griffin_lim:
        waves['vocoder'] = waves['inv_transforms'][:int(22050 * _L)]
    else:
        waves['vocoder'] = waves['vocoder'][:int(22050 * _L)]
    cond_save_path = os.path.join(target_cond_v_dir, Path(ref_video_path).stem +
                                  '_to_' + Path(_cond_video_path).stem + '.wav')
    soundfile.write(cond_save_path, cond_waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
    print(f'The sample has been saved @ {cond_save_path}')
    cond_save_video_path = os.path.join(target_cond_v_dir, Path(ref_video_path).stem +
                                        '_to_' + Path(_cond_video_path).stem + '.mp4')
    attach_audio_to_video(_cond_video_path, cond_save_path, cond_save_video_path, 0, recon_only=True, v_duration=_L)

    # plot melspec
    # target
    plt.imshow(mel_xrec[0, 0, :, :spec_take_first], cmap='coolwarm', origin='lower')
    plt.axis('off')
    plt.savefig(orig_save_video_path.replace('.mp4', '.jpg'), bbox_inches='tight', pad_inches=0.)
    plt.close()

    # condition
    _spec_take_first = int(spec_take_first // W_scale) if fix_cond else spec_take_first
    plt.imshow(mel_xprec[0, 0, :, :_spec_take_first], cmap='coolwarm', origin='lower')
    plt.axis('off')
    plt.savefig(cond_save_video_path.replace('.mp4', '.jpg'), bbox_inches='tight', pad_inches=0.)
    plt.close()

    # generated
    draw_spec(mel_z[0, 0, :, :spec_take_first], save_video_path.replace('.mp4', '.jpg'), cmap='coolwarm')
    return


# ------------------------------
# Multiple Outputs (re-ranking)
# ------------------------------
def gen_audio_condImage_fast_multiple(
    video_path,
    cond_video_path,
    model,
    all_gen_dict,
    target_log_dir='CondAVTransformer',
    SR=22050,
    FPS=15,
    L=2.0,
    normalize=False,
    using_torch=False,
    show_griffin_lim=False,
    vqgan_L=10.0,
    style_transfer=False,
    target_start_time=0,
    cond_start_time=0,
    outside=False,
    remove_noise=False,
    spec_take_first=160,
    gen_cnt=25,
    W_scale=1,
    slide_win_mode='half',
    temperature=1.0,
    ignore_input_spec=False,
    tmp_path='./tmp',
    fix_cond=True
):
    pass
#     '''
#     parameters:
#         video_path: path to the target video, will be trimmed to 2s and re-encode into 15 fps.
#         cond_video_path: path to the conditional video, will be trimmed to 2s and re-encode into 15 fps.
#         model: model object, returned by load_model function
#         target_log_dir: target output dir name in the 'logs' directory, e.g. output will be saved to 'logs/<target_log_dir>'
#         SR: sampling rate
#         FPS: Frame rate
#         L: length of generated sound
#         normalize: whether to normaliza input waveform
#         using_torch: use torchaudio to extrac spectrogram
#         show_griffin_lim: use griffin_lim algorithm vocoder
#         vqgan_L: length of VQ-GAN codebook, use 2 if using GreatestHit codebook
#         style_transfer: generate style transfer sound
#         target_start_time: if target video is from outside, trim from <target_start_time> to <target_start_time>+2
#         cond_start_time: if conditional video is from outside, trim from <cond_start_time> to <cond_start_time>+2
#         outside: indicate whether the video from outside source
#         remove_noise: denoise for outside videos
#         spec_take_first: size of the spectrogram to use
#         gen_cnt: count of generation times
#         W_scale: scale of audio duration as multiples of 2sec
#         slide_win_mode: mode of sliding window, choose from ['half', 'last']
#         temperature: temperature of multinomial sampling.
#         ignore_input_spec: ignore input spec when input video is silent
#         tmp_path: tmp dir to save intermediate files
#         fix_cond: use only 2 sec condition regardless to input length.
#     '''

#     config, sampler, melgan, melception = model
#     # feature extractor
#     L = int(L * W_scale)
#     vqgan_L = int(vqgan_L * W_scale)
#     spec_take_first = int(spec_take_first * W_scale)
#     if '_denoised_' not in video_path or outside:
#         new_fps_video_path = reencode_video_with_diff_fps(video_path, tmp_path, FPS)
#         video_path = trim_video(new_fps_video_path, target_start_time, vqgan_L, tmp_path=tmp_path)
#     frames = [Image.fromarray(f) for f in load_frames(video_path)][:int(FPS * L)]
#     frames = FRAME_TRANS(frames)

#     if '_denoised_' not in cond_video_path or outside:
#         new_fps_video_path = reencode_video_with_diff_fps(cond_video_path, tmp_path, FPS)
#         cond_video_path = trim_video(new_fps_video_path, cond_start_time, vqgan_L, tmp_path=tmp_path)
#     cond_frames = [Image.fromarray(f) for f in load_frames(cond_video_path)][:int(FPS * L)]
#     cond_frames = FRAME_TRANS(cond_frames)

#     feats = {'feature': np.stack(cond_frames + frames, axis=0)}

#     cond_video_path = cond_video_path
#     ref_video_path = video_path

#     # Extract Features
#     visual_features = feats

#     # Prepare Input
#     batch = default_collate([visual_features])
#     batch['feature'] = batch['feature'].to(device)
#     with torch.no_grad():
#         c = sampler.get_input(sampler.cond_stage_key, batch)

#     if not ignore_input_spec:
#         # Extract Spectrogram
#         spectrogram = extract_melspectrogram(ref_video_path, SR, normalize=normalize, using_torch=using_torch, remove_noise=remove_noise, duration=vqgan_L, tmp_path=tmp_path)
#         spec_H, spec_W = spectrogram.shape
#         if spec_W > spec_take_first:
#             spectrogram = spectrogram[:, :spec_take_first]
#         else:
#             pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
#             pad[:, :spec_W] = spectrogram
#             spectrogram = pad
#         spectrogram = {'input': spectrogram}
#         if config.data.params.spec_crop_len is None or W_scale != 1:
#             config.data.params.spec_crop_len = spec_take_first
#         if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
#             random_crop = False
#             crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
#             spectrogram = crop_img_fn(spectrogram)

#         # Prepare input
#         batch = default_collate([spectrogram])
#         batch['image'] = batch['input'].to(device)
#         x = sampler.get_input(sampler.first_stage_key, batch)
#         mel_x = x.detach().cpu().numpy()

#         # Encode and Decode the Spectrogram
#         with torch.no_grad():
#             quant_z, z_indices = sampler.encode_to_z(x)
#             # print(z_indices)
#             xrec = sampler.first_stage_model.decode(quant_z)
#             mel_xrec = xrec.detach().cpu().numpy()

#     # Conditional
#     # Extract Spectrogram
#     spectrogram = extract_melspectrogram(cond_video_path, SR, normalize=normalize, using_torch=using_torch, remove_noise=remove_noise, duration=vqgan_L, tmp_path=tmp_path)
#     spec_H, spec_W = spectrogram.shape
#     if spec_W > spec_take_first:
#         spectrogram = spectrogram[:, :spec_take_first]
#     else:
#         pad = np.zeros((spec_H, spec_take_first), dtype=spectrogram.dtype)
#         pad[:, :spec_W] = spectrogram
#         spectrogram = pad
#     spectrogram = {'input': spectrogram}
#     if config.data.params.spec_crop_len is None or W_scale != 1:
#         config.data.params.spec_crop_len = spec_take_first
#     if spectrogram['input'].shape[1] > config.data.params.spec_crop_len:
#         random_crop = False
#         crop_img_fn = CropImage([config.data.params.mel_num, config.data.params.spec_crop_len], random_crop)
#         spectrogram = crop_img_fn(spectrogram)

#     # Prepare input
#     batch = default_collate([spectrogram])
#     batch['cond_image'] = batch['input'].to(device)
#     xp = sampler.get_input(sampler.cond_first_stage_key, batch)
#     mel_xp = xp.detach().cpu().numpy()

#     # Encode and Decode the Spectrogram
#     with torch.no_grad():
#         quant_zp, zp_indices = sampler.encode_to_z(xp)
#         # print(zp_indices)
#         xprec = sampler.first_stage_model.decode(quant_zp)
#         mel_xprec = xprec.detach().cpu().numpy()

#     if ignore_input_spec:
#         z_indices = torch.zeros_like(zp_indices)
#         xrec = torch.zeros_like(xprec)
#         mel_xrec = np.zeros_like(mel_xprec)

#     # Define Sampling Parameters
#     # take top 1024 / 512 code
#     top_x = sampler.first_stage_model.quantize.n_e // 2

#     if not os.path.exists(f'logs/{target_log_dir}'):
#         os.mkdir(f'logs/{target_log_dir}')

#     if video_path not in all_gen_dict.keys():
#         all_gen_dict[video_path] = {}
#     all_gen_dict[video_path][cond_video_path] = []
#     # Start sampling
#     if style_transfer:
#         content_img = load_specs_as_img(mel_xrec[0, 0, :, :spec_take_first])
#         style_img = load_specs_as_img(mel_xprec[0, 0, :, :spec_take_first])
#         generated_spec = run_style_transfer(
#             cnn_normalization_mean.to(),
#             cnn_normalization_std.to(),
#             content_img.clone().to(device),
#             style_img.clone().to(device),
#             content_img.clone().to(device),
#         )
#         z_pred_img = torch.mean(generated_spec, dim=1, keepdim=True)
#         mel_z = z_pred_img.detach().cpu().numpy()
#     else:
#         for _ in range(gen_cnt):
#             with torch.no_grad():
#                 start_t = time.time()

#                 quant_c, c_indices = sampler.encode_to_c(c)
#                 z_indices_clip = z_indices[:, :sampler.clip * W_scale]
#                 zp_indices_clip = zp_indices[:, :sampler.clip * W_scale]
#                 z_indices_rec = z_indices.clone()
#                 # crec = sampler.cond_stage_model.decode(quant_c)

#                 patch_size_i = 5

#                 c_window_size = int(2 * FPS)

#                 # TODO: modify the shape if drop condition info
#                 downsampled_size = spec_take_first // 16
#                 cond_patch_shift_j = (W_scale - 1) * (downsampled_size // W_scale)
#                 if 'dropcond_' in target_log_dir:
#                     B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(downsampled_size))
#                     patch_size_j = int(downsampled_size // W_scale)
#                 else:
#                     B, D, hr_h, hr_w = sampling_shape = (1, 256, 5, int(2*downsampled_size))
#                     patch_size_j = int(2*downsampled_size // W_scale)
#                 z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(device)

#                 if 'dropcond_' not in target_log_dir:
#                     start_step = zp_indices_clip.shape[1]
#                     z_pred_indices[:, :start_step] = zp_indices_clip[:, :start_step]
#                 elif 'dropcond_' in target_log_dir:
#                     start_step = 0

#                 for step in range(start_step, hr_w * hr_h):
#                     i = step % hr_h
#                     j = step // hr_h

#                     i_start = min(max(0, i - (patch_size_i // 2)), hr_h - patch_size_i)
#                     if slide_win_mode == 'half':
#                         j_start = min(max(0, j - (3 * patch_size_j // 4)), hr_w - patch_size_j)
#                     elif slide_win_mode == 'last':
#                         j_start = min(max(0, j - patch_size_j + 1), hr_w - patch_size_j)
#                     else:
#                         raise NotImplementedError
#                     i_end = i_start + patch_size_i
#                     j_end = j_start + patch_size_j

#                     local_i = i - i_start

#                     patch_2d_shape = (B, D, patch_size_i, patch_size_j)

#                     if W_scale != 1:
#                         # if fix cond, we always use first 2 sec of cond audio.
#                         cond_j_start = 0 if fix_cond else max(0, j_start - cond_patch_shift_j)
#                         cond_j_end = cond_j_start + (patch_size_j // 2)
#                         tar_j_start = max(0, j_start - cond_patch_shift_j) + (hr_w // 2)
#                         tar_j_end = tar_j_start + (patch_size_j // 2)

#                         local_j = j - tar_j_start + (patch_size_j // 2)

#                         cond_patch = z_pred_indices \
#                             .reshape(B, hr_w, hr_h) \
#                             .permute(0, 2, 1)[:, i_start:i_end, cond_j_start:cond_j_end].permute(0, 2, 1)
#                         tar_patch = z_pred_indices \
#                             .reshape(B, hr_w, hr_h) \
#                             .permute(0, 2, 1)[:, i_start:i_end, tar_j_start:tar_j_end].permute(0, 2, 1)
#                         patch = torch.cat([cond_patch, tar_patch], dim=1).reshape(B, patch_size_i * patch_size_j)

#                         cond_t_start = cond_j_start * 0.2
#                         cond_frame_start = int(cond_t_start * FPS)
#                         cond_frame_end = cond_frame_start + c_window_size
#                         tar_frame_start = int(cond_frame_start + c_window_size * W_scale)
#                         tar_frame_end = tar_frame_start + c_window_size
#                         cpatch = torch.cat([c_indices[:, :, cond_frame_start:cond_frame_end], c_indices[:, :, tar_frame_start:tar_frame_end]], dim=2)
#                     else:
#                         local_j = j - j_start
#                         patch = z_pred_indices
#                         # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
#                         cpatch = c_indices

#                     logits, _, attention = sampler.transformer(patch[:, :-1], cpatch)
#                     # remove conditioning
#                     logits = logits[:, -patch_size_j*patch_size_i:, :]

#                     local_pos_in_flat = local_j * patch_size_i + local_i
#                     logits = logits[:, local_pos_in_flat, :]

#                     logits = logits / temperature
#                     logits = sampler.top_k_logits(logits, top_x)

#                     # apply softmax to convert to probabilities
#                     probs = torch.nn.functional.softmax(logits, dim=-1)

#                     # sample from the distribution
#                     ix = torch.multinomial(probs, num_samples=1)
#                     z_pred_indices[:, j * hr_h + i] = ix

#                 # quant_z_shape = sampling_shape
#                 if 'dropcond_' in target_log_dir:
#                     z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices
#                 else:
#                     z_indices_rec[:, :sampler.clip * W_scale] = z_pred_indices[:, sampler.clip * W_scale:]
#                 # print(z_indices_rec)
#                 z_pred_img = sampler.decode_to_img(z_indices_rec,
#                                                    (1, 256, 5, 53 if vqgan_L == 10.0 else downsampled_size))
#                 mel_z = z_pred_img.detach().cpu().numpy()
#                 config.data.params.spec_dir_path = 'melspec_10s_22050hz'
#                 waves = spec_to_audio_to_streamlit(z_pred_img, config.data.params.spec_dir_path,
#                                             config.data.params.sample_rate, show_griffin_lim=show_griffin_lim,
#                                             vocoder=melgan, show_in_st=False)
#                 waves['vocoder'] = waves['vocoder'][:int(22050 * L)]
#                 all_gen_dict[video_path][cond_video_path].append(waves['vocoder'])

#     return
