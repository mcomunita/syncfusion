import argparse
import os
import json

from glob import glob
from natsort import natsorted

def process_annotation(
    times_path,
    times_suffix,
    output_dir,
    flatten
):
    times_name = os.path.basename(times_path).replace(times_suffix, '')
    if flatten:
        times_output_path = output_dir
    else:
        times_output_path = os.path.join(output_dir, times_name)
    
    # make output directory
    os.makedirs(times_output_path, exist_ok=True)
    # print(video_output_path)

     # read file
    with open(times_path, 'r') as f:
        lines = f.read().splitlines()

    # split each line into time and labels
    for i, line in enumerate(lines):
        time = line.split(' ')[0]
        labels = ' '.join(str(e) for e in line.split(' ')[1:])
        lines[i] = [time, labels]
    
    # write to csv
    with open(os.path.join(times_output_path, f"{times_name}.times.csv"), 'w') as f:
        for line in lines:
            f.write(f"{line[0]},{line[1]}\n")

    # write to json
    # with open(os.path.join(times_output_path, f"hit_record.json"), 'w') as f:
    #     f.write(str(lines))

    # write to json
    # json_object = json.dumps(lines, indent=4)
    # with open(os.path.join(times_output_path, f"hit_record.json"), "w") as f:
    #     f.write(json_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test on 5 videos")
    parser.add_argument("--flatten", action="store_true", help="Flatten directory structure")
    parser.add_argument("-it", "--input_dir_times", default='/Volumes/STEVE/DATASETS/GREATEST-HITS-NEW/times-txt')
    parser.add_argument("-ts", "--times_suffix", default="_times.txt")
    parser.add_argument("-o", "--output_dir", default='/Volumes/STEVE/DATASETS/GREATEST-HITS-NEW/mic-mp4-processed')
    args = parser.parse_args()

    input_dir_times = args.input_dir_times
    times_suffix = args.times_suffix
    output_dir = args.output_dir

    times_paths = natsorted(glob(os.path.join(input_dir_times, f"*{times_suffix}")))
    if args.test:
        times_paths = times_paths[:5] # for testing

    for t in times_paths:
        process_annotation(
            times_path=t,
            times_suffix=times_suffix,
            output_dir=output_dir,
            flatten=args.flatten
        )
    
    print("Done!")
