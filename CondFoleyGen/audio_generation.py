import argparse
import json
import os
import random
from pprint import pprint

import torch

from tqdm import tqdm

from specvqgan.data.datamodule_greatesthits import CondGreatestHitsWaveCondOnImageDatamodule
from feature_extraction.demo_utils import load_model
from audio_generation_scripts import \
    gen_audio_condImage_fast, \
    gen_audio_condImage_fast_multiple, \
    generate_audio_using_ref_video_and_cond_audiovideo


parser = argparse.ArgumentParser()
# parser.add_argument('--model_name', type=str, default=None, help='folder name of the pre-trained model')
parser.add_argument('--transformer_ckpt_path', type=str, default=None, help='path to transformer ckpt file')
parser.add_argument('--transformer_config_path', type=str, default=None, help='path to transformer config file')
parser.add_argument('--target_log_dir', type=str, default=None, help='output folder name under logs/ dir')
parser.add_argument('--temp_dir', type=str, help='temporary folder to place intermediate result')
parser.add_argument('--test_data_to_use', type=float, default=1.0, help='ratio of test data to use')
parser.add_argument('--test_chunks_to_use', type=float, default=1.0, help='ratio of test chunks to use')
parser.add_argument('--shuffle_test_chunks', action='store_true', help='shuffle test chunks')
parser.add_argument('--chunks_to_generate', type=int, default=-1, help='number of output chunks to generate')


parser.add_argument('--orig_videos_dir', type=str, default=None, help='folder for original unprocessed videos')
parser.add_argument('--orig_videos_suffix', type=str, default="_mic.mp4", help='suffix for original unprocessed videos')

parser.add_argument('--slide_win_mode', type=str, default='half', choices=['half', 'last'], help='slide window method when generating longer audio')
parser.add_argument('--W_scale', type=int, default=1, help='length scale of the generate audio, output will be W_scale*2s')
# parser.add_argument('--max_W_scale', type=int, default=3, help='maximum W_scale to iterate')
# parser.add_argument('--min_W_scale', type=int, default=1, help='minimum W_scale to iterate')
# parser.add_argument('--gen_cnt', type=int, default=30, help='generation count when generating multiple result')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature of softmax for logits to probability')
# parser.add_argument('--split', type=int, default=-1, help='split idx when running multi-process generation')
# parser.add_argument('--total_split', type=int, default=1, help='total number of multi-process')
# parser.add_argument('--tmp_idx', type=int, default=-1, help='temperate folder idx to place intermediate result')

parser.add_argument('--new_codebook', action='store_true', help='load a different codebook according to config')
parser.add_argument('--gh_testset', action='store_true', help='running greatest hit testset')
# parser.add_argument('--gh_demo', action='store_true', help='running the greatest hit demo')
# parser.add_argument('--gh_gen', action='store_true', help='generate audio with greatest hit model')
# parser.add_argument('--countix_av_gen', action='store_true', help='generate audio with countix-AV model')
parser.add_argument('--multiple', action='store_true', help='generate multiple audio for each pair of input for re-ranking')


# create a module to normalize input image so we can easily put it in a
# nn.Sequential


if __name__ == '__main__':
    args = parser.parse_args()
    pprint(args)

    random.seed(1234) # change seed to videos with different conditionings

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.transformer_ckpt_path
    config_path = args.transformer_config_path
    target_log_dir = args.target_log_dir
    temp_dir = args.temp_dir
    new_codebook = args.new_codebook
    slide_win_mode = args.slide_win_mode

    # load tranformer and vocoder from checkpoint
    config, sampler, vocoder = load_model(
        model_path,
        config_path,
        device,
        load_feat_extractor=False,
        load_new_first_stage=new_codebook
    )

    print("\nCONFIG")
    pprint(config)

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(target_log_dir, exist_ok=True)

    if args.gh_testset:

        # setup datamodule and dataloader
        print("\nSETUP GH DATASET")

        if args.W_scale > 1: # need to increase chunk length in the dataset
            config["data"]["init_args"]["chunk_length_in_seconds"] = \
                config["data"]["init_args"]["chunk_length_in_seconds"] * args.W_scale

        # limit number of test videos
        if args.test_data_to_use < 1.0:
            print(f"Using {args.test_data_to_use} of test data")
            config["data"]["init_args"]["test_data_to_use"] = args.test_data_to_use

        datamodule = CondGreatestHitsWaveCondOnImageDatamodule(
            **config["data"]["init_args"]
        )
        datamodule.setup("test")
        test_dataset = datamodule.test_dataset

        # shuffle test chunks
        if args.shuffle_test_chunks:
            print("Shuffling test chunks")
            random.shuffle(test_dataset.list_onsets)

        # limit number of test chunks
        if args.test_chunks_to_use < 1.0:
            print(f"Using {args.test_chunks_to_use} of test chunks")
            test_dataset.list_onsets = \
                test_dataset.list_onsets[:int(args.test_chunks_to_use * len(test_dataset))]
        
        # limit number of chunks to generate
        if args.chunks_to_generate > 0:
            print(f"Generating {args.chunks_to_generate} chunks")
            n_gen_chunks = args.chunks_to_generate
        else:
            n_gen_chunks = len(test_dataset)
        
        print(f"\nNumber of test videos: {len(test_dataset.list_samples)}")
        print(f"Number of test chunks: {len(test_dataset.list_onsets)}")
        
        # make sure dataset is set to return a conditioning video
        # that is different from the reference video
        if test_dataset.p_outside_cond != 1.0:
            print("Setting p_outside_cond to 1.0")
            test_dataset.p_outside_cond = 1.0

        print("\nGenerating...")
        if args.multiple:
            pass
        else:
            for i in range(n_gen_chunks):
                print(f"\n{i}")
                item = test_dataset[i]
                generate_audio_using_ref_video_and_cond_audiovideo(
                    config,
                    sampler,
                    vocoder,
                    item,
                    target_log_dir,
                    temp_dir,
                    original_video_dir=args.orig_videos_dir,
                    original_video_suffix=args.orig_videos_suffix,
                    sample_rate=test_dataset.sample_rate,
                    frame_rate=test_dataset.frame_rate,
                    chunk_length_in_seconds=test_dataset.chunk_length_in_seconds,
                    vqgan_codebook_length_in_seconds=test_dataset.chunk_length_in_seconds,
                    normalize=False,
                    using_torch=True,
                    show_griffin_lim=False,
                    n_spectrogram_time_frames=160,
                    W_scale=args.W_scale,
                    fix_cond_length=True,
                    device=device,
                )
        #     # generate audio for each pair of ref and cond video
        #     for i, (ref_video_path, cond_video_paths) in enumerate(test_dataset):
        #         start_sec = None
        #         for j, cvp in enumerate(cond_video_paths):
        #             gen_audio_condImage_fast(
        #                 ref_video_path,
        #                 cvp,
        #                 model,
        #                 spec_take_first=160,
        #                 target_log_dir=target_log_dir,
        #                 using_torch=True,
        #                 L=2.0,
        #                 cond_cnt=j,
        #                 style_transfer=False,
        #                 normalize=False,
        #                 show_griffin_lim=False,
        #                 vqgan_L=2.0,
        #                 W_scale=args.W_scale
        #             )

        # with re-ranking
        if args.multiple:
            # # generate 100 for single bad example.
            # split = args.split
            # num_splits = args.total_split
            # all_gen_dict = {}
            # gen_cnt = 100
            # dest = os.path.join(f'logs/{target_log_dir}', f'{gen_cnt}_times_split_{split}_wav_dict.pt')

            # for i, (v, extra_video_paths) in enumerate(tqdm(fixed_test.items())):
            #     video_path = v
            #     start_sec = None
            #     if i % num_splits != split:
            #         continue
            #     for j, ep in enumerate(extra_video_paths):
            #         gen_audio_condImage_fast_multiple(
            #             video_path,
            #             ep,
            #             model,
            #             all_gen_dict,
            #             spec_take_first=160,
            #             gen_cnt=gen_cnt,
            #             target_log_dir=target_log_dir,
            #             using_torch=True,
            #             L=2.0,
            #             cond_cnt=j,
            #             style_transfer=False,
            #             normalize=False,
            #             show_griffin_lim=False,
            #             vqgan_L=2.0
            #         )
            # torch.save(all_gen_dict, dest)
            pass
        else:
            # test set example:
            # "2015-02-21-17-09-17_3.0667": ["2015-10-06-20-27-19-220_7.5333", "2015-02-16-17-52-15_11.5333", "2015-10-06-19-28-22-1_9.6000"]
            # for each reference video, there are 3 cond. videos to test on
            # for i, (ref_video_path, cond_video_paths) in enumerate(fixed_test.items()):
            #     start_sec = None
            #     for j, cvp in enumerate(cond_video_paths):
            #         gen_audio_condImage_fast(
            #             ref_video_path,
            #             cvp,
            #             model,
            #             spec_take_first=160,
            #             target_log_dir=target_log_dir,
            #             using_torch=True,
            #             L=2.0,
            #             cond_cnt=j,
            #             style_transfer=False,
            #             normalize=False,
            #             show_griffin_lim=False,
            #             vqgan_L=2.0,
            #             W_scale=args.W_scale
            #         )
            pass
    else:
        raise NotImplementedError
