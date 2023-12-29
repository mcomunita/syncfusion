import argparse
import os
import soundfile as sf
import noisereduce
import numpy as np
import pandas as pd
import ffmpeg
import json

from glob import glob
from multiprocessing import Pool
from functools import partial
from natsort import natsorted


def pipeline(
    video_path,
    video_suffix,
    audio_sample_rate,
    audio_bitdepth,
    audio_denoise,
    audio_onsets,
    video_frames_per_second,
    video_width,
    video_height,
    output_dir,
    flatten
):
    video_name = os.path.basename(video_path).replace(video_suffix, '')
    print("Processing:", video_name)

    # make output directory
    if flatten:
        video_output_path = output_dir
    else:
        video_output_path = os.path.join(output_dir, video_name)
    os.makedirs(video_output_path, exist_ok=True)

    # save metadata
    metadata_in = ffmpeg.probe(video_path)
    metadata_out = {
        "original": {
            "width": int(metadata_in["streams"][0]["width"]),
            "height": int(metadata_in["streams"][0]["height"]),
            "video_frame_rate": float(eval(metadata_in["streams"][0]["avg_frame_rate"])),
            "video_duration": float(metadata_in["streams"][0]["duration"]),
            "video_num_frames": int(metadata_in["streams"][0]["nb_frames"]),
            "audio_sample_rate": int(metadata_in["streams"][1]["sample_rate"]),
            "audio_channels": int(metadata_in["streams"][1]["channels"]),
            "audio_duration": float(metadata_in["streams"][1]["duration"]),
            "audio_num_samples": int(metadata_in["streams"][1]["duration_ts"]),
        },
        "processed": {
            "width": int(video_width),
            "height": int(video_height),
            "video_frame_rate": int(video_frames_per_second),
            "video_duration": float(metadata_in["streams"][0]["duration"]),
            "video_num_frames": int(float(metadata_in["streams"][0]["duration"]) * video_frames_per_second),
            "audio_sample_rate": int(audio_sample_rate),
            "audio_channels": 1,
            "audio_bitdepth": int(audio_bitdepth),
        }
    }

    with open(os.path.join(video_output_path, f'{video_name}.metadata.json'), 'w') as f:
        json.dump(metadata_out, f, indent=4)

    # extract audio
    if flatten:
        audio_output_dir = output_dir
    else:
        audio_output_dir = os.path.join(video_output_path, "audio")  # audio subdir
    os.makedirs(audio_output_dir, exist_ok=True)

    audio_output_path = os.path.join(audio_output_dir, f'{video_name}.resampled.wav')

    if audio_bitdepth == 32:  # float32
        ffmpeg_format = 'pcm_f32le'
        sf_format = 'FLOAT'
    elif audio_bitdepth == 24:  # int24
        ffmpeg_format = 'pcm_s24le'
        sf_format = 'PCM_24'
    elif audio_bitdepth == 16:  # int16
        ffmpeg_format = 'pcm_s16le'
        sf_format = 'PCM_16'
    else:
        raise ValueError(f"Audio bitrate {audio_bitdepth} not supported. Please choose from 32, 24, 16.")

    os.system(f"ffmpeg -i {video_path} -loglevel error -ar {audio_sample_rate} -ac 1 -c:a {ffmpeg_format} -y {audio_output_path}")

    # denoise resampled audio
    if audio_denoise:
        audio_input_path = audio_output_path
        audio_denoised_output_path = os.path.join(audio_output_dir, f'{video_name}.resampled_denoised.wav')
        x, sr = sf.read(audio_input_path)
        if len(x.shape) == 1:
            x = x[None, :]
        x_denoised = noisereduce.reduce_noise(x, sr, n_fft=1024, hop_length=1024//4)
        x_denoised = x_denoised.squeeze()
        sf.write(audio_denoised_output_path, x_denoised, samplerate=sr, subtype=sf_format)

    # generate onset track
    if audio_onsets:
        audio_input_path = audio_output_path
        audio_onset_output_path = os.path.join(audio_output_dir, f'{video_name}.resampled_onset.wav')
        x, sr = sf.read(audio_input_path)
        annotations_path = os.path.join(video_output_path, "hit_record.csv")
        df = pd.read_csv(annotations_path, header=None, names=['hits', 'labels'])
        onsets_in_seconds = df['hits'].values
        onsets_in_samples = (onsets_in_seconds * sr).astype(int)
        onset_track = np.zeros_like(x)
        onset_track[onsets_in_samples] = 1
        sf.write(audio_onset_output_path, onset_track, samplerate=sr, subtype=sf_format)

    # extract frames
    if flatten:
        frames_output_dir = output_dir
    else:
        frames_output_dir = os.path.join(video_output_path, "frames")  # frames subdir
    os.makedirs(frames_output_dir, exist_ok=True)

    frames_output_path = os.path.join(frames_output_dir, f"{video_name}.frame_%06d.jpg")
    os.system(f'ffmpeg -i {video_path} -loglevel error -filter:v fps=fps={video_frames_per_second},scale={video_width}:{video_height} -y {frames_output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test on 5 videos")
    parser.add_argument("--flatten", action="store_true", help="Flatten directory structure")

    parser.add_argument("-iv", "--input_dir_videos", default='/Volumes/STEVE/DATASETS/GREATEST-HITS-NEW/mic-mp4')
    parser.add_argument("-vs", "--videos_suffix", default="_mic.mp4")

    parser.add_argument("-o", "--output_dir", default='/Volumes/STEVE/DATASETS/GREATEST-HITS-NEW/mic-mp4-processed')
    parser.add_argument("-nw", "--num_workers", type=int, default='8')

    parser.add_argument("-asr", '--audio_sample_rate', type=int, default='48000')
    parser.add_argument("-abd", '--audio_bitdepth', type=int, default='32')
    parser.add_argument("-adn", '--audio_denoise', action='store_true', default=False)
    parser.add_argument("-ao", '--audio_onsets', action='store_true', default=False)

    parser.add_argument("-vfps", '--video_fps', type=int, default='15')
    parser.add_argument("-vw", '--video_width', type=int, default='320')
    parser.add_argument("-vh", '--video_height', type=int, default='240')

    args = parser.parse_args()

    input_dir_videos = args.input_dir_videos
    videos_suffix = args.videos_suffix

    output_dir = args.output_dir
    num_workers = args.num_workers

    audio_sample_rate = args.audio_sample_rate
    audio_bitdepth = args.audio_bitdepth
    audio_denoise = args.audio_denoise
    audio_onsets = args.audio_onsets

    video_fps = args.video_fps
    video_width = args.video_width
    video_height = args.video_height

    videos_paths = natsorted(glob(os.path.join(input_dir_videos, f"*{videos_suffix}")))
    if args.test:
        videos_paths = videos_paths[:5]  # for testing

    with Pool(num_workers) as p:
        p.map(partial(
            pipeline,
            video_suffix=videos_suffix,
            audio_sample_rate=audio_sample_rate,
            audio_bitdepth=audio_bitdepth,
            audio_denoise=audio_denoise,
            audio_onsets=audio_onsets,
            video_frames_per_second=video_fps,
            video_width=video_width,
            video_height=video_height,
            output_dir=output_dir,
            flatten=args.flatten
        ),
            videos_paths
        )

    print("Done!")
