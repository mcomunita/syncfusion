import argparse
import os
import random
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-iv", "--input_dir_videos", default='/Volumes/STEVE/DATASETS/GREATEST-HITS-NEW/mic-mp4')
    parser.add_argument("-vs", "--videos_suffix", default="_mic.mp4")
    parser.add_argument("-o", "--output_dir", default='/Volumes/STEVE/DATASETS/GREATEST-HITS-NEW')
    parser.add_argument("-trs", '--train_split_ratio', type=float, default='0.7')
    parser.add_argument("-vls", '--val_split_ratio', type=float, default='0.1')
    parser.add_argument("-tss", '--test_split_ratio', type=float, default='0.2')
    args = parser.parse_args()

    train_split = args.train_split_ratio
    val_split = args.val_split_ratio
    test_split = args.test_split_ratio

    assert train_split + val_split + test_split == 1.0, "Split ratios must sum to 1.0"

    os.makedirs(args.output_dir, exist_ok=True)

    input_dir_videos = args.input_dir_videos
    videos_suffix = args.videos_suffix

    videos_paths = glob(os.path.join(input_dir_videos, f"*{videos_suffix}"))
    videos_names = [os.path.basename(v).replace(videos_suffix, '') for v in videos_paths]

    # shuffle
    random.seed(42)
    random.shuffle(videos_names)

    # split
    train_split_idx = int(len(videos_names) * train_split)
    val_split_idx = int(len(videos_names) * (train_split + val_split))

    train_videos = sorted(videos_names[:train_split_idx])
    val_videos = sorted(videos_names[train_split_idx:val_split_idx])
    test_videos = sorted(videos_names[val_split_idx:])

    # write to file
    with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
        for v in train_videos:
            f.write(f"{v}\n")
    
    with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
        for v in val_videos:
            f.write(f"{v}\n")
        
    with open(os.path.join(args.output_dir, 'test.txt'), 'w') as f:
        for v in test_videos:
            f.write(f"{v}\n")
    
    print(f"Train: {len(train_videos)}")
    print(f"Val: {len(val_videos)}")
    print(f"Test: {len(test_videos)}")
    print("Done!")