import os
import numpy as np
import json
import librosa
import argparse
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

from glob import glob

##############################################################################
# EVALUATE ACC & AP COMPLETE MODEL
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--gen_dir', type=str)
parser.add_argument('--tar_dir', type=str, default='data/AMT_test/target_sound')
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--remove_head', type=float, default=None)
parser.add_argument('--plt', action='store_true')
parser.add_argument('--multi_delta', action='store_true')
parser.add_argument('--longer_det', action='store_true')
args = parser.parse_args()

def load_wav_audio(audio_path):
    return librosa.load(audio_path, sr=22050)[0]

def detect_onset(audio_dir, duration=2):
    audio_list = glob(os.path.join(audio_dir, '*.wav'))
    audio_list.sort()
    onset_res = {}
    wav_res = {}
    for audio_path in tqdm(audio_list):
        audio_name = os.path.basename(audio_path)
        
        onset_res[audio_name] = []
        wav = load_wav_audio(audio_path)
        onsets = librosa.onset.onset_detect(y=wav, sr=22050, units='samples', delta=0.3)
        onset_res[audio_name] = onsets
        wav_res[audio_name] = wav
    return onset_res, wav_res

def onset_nms(onset, confidence, window=0.05):
    onset_remain = onset.tolist()
    output = []
    sorted_idx = np.argsort(confidence)[::-1]
    for idx in sorted_idx:
        cur = onset[idx]
        if cur not in onset_remain:
            continue
        output.append(cur)
        onset_remain.remove(cur)
        for o in onset_remain:
            if abs(cur - o) < window * 22050:
                onset_remain.remove(o)
    return np.array(sorted(output))


def eval_osnets(onset1, onset2, wav2, delta=0.1, conf_interval=int(0.05*22050), keys=None):
    wav2 = np.abs(wav2)
    wav2 = (wav2 - np.min(wav2)) / (np.max(wav2) - np.min(wav2))
    confidence = [np.max(wav2[o-conf_interval:o+conf_interval]) for o in onset2]
    onset2 = onset_nms(onset2, confidence)
    onset2_keep = onset2
    onset2_onuse = copy.deepcopy(onset2.tolist())
    onset2_res = [0 for _ in onset2_onuse]
    hit_cnt = 0
    y_gt = []
    y_pred = []
    for o in onset1:
        diff = [abs(o2 - o) for o2 in onset2_onuse]
        idx_in_window = [idx for idx in range(len(onset2_onuse)) if diff[idx] < delta * 22050]
        if len(idx_in_window) == 0:
            y_gt.append(1)
            y_pred.append(0)
        else:
            conf_in_window = [wav2[onset2_onuse[idx]] for idx in idx_in_window]
            max_conf_idx = np.argsort(conf_in_window)[-1]
            match_idx = idx_in_window[max_conf_idx]
            hit_cnt += 1
            y_gt.append(1)
            conf = np.max(wav2[onset2_onuse[match_idx]-conf_interval:onset2_onuse[match_idx]+conf_interval])
            for i in range(len(onset2_keep)):
                if onset2_keep[i] == onset2_onuse[match_idx]:
                    onset2_res[i] = 1
            y_pred.append(conf)
            onset2_onuse.remove(onset2_onuse[match_idx])
            if len(onset2_onuse) == 0:
                break
    for o in onset2_onuse:
        y_gt.append(0)
        y_pred.append(np.max(wav2[o-conf_interval:o+conf_interval]))
    acc = hit_cnt / len(onset1) if len(onset1) != 0 else 0
    ap = average_precision_score(y_gt, y_pred)
    pr, rc, th = precision_recall_curve(y_gt, y_pred)
    if keys != None and args.plt:
        plt.plot(rc, pr)
        plt.ylim((0, 1))
        plt.savefig(f'/homes/emilpost/riccardo/diffusion-sfx/temp_aug/pc_rc_curve_tar_{keys[0]}_cond_{keys[1]}.jpg')
        plt.close()
    return acc, ap, onset2_res

def plot_onset(path, wav1, wav2, onsets1, onsets2, pred=None, ap=None, conf_interval=int(0.05*22050)):
    plt.subplot(2, 1, 1)
    conf = np.abs(wav1)
    conf = (conf - np.min(conf)) / (np.max(conf) - np.min(conf))
    plt.plot(np.arange(len(wav1)), wav1)
    for idx, x in enumerate(onsets1):
        confidence = np.max(conf[x-conf_interval:x+conf_interval])
        plt.text(x+10, np.min(wav1) - 0.2, f"conf: {confidence:.2f}")
        plt.vlines(x, np.min(wav1) - 0.2, np.max(wav1) + 0.2, colors='black')

    plt.subplot(2, 1, 2) 
    conf = np.abs(wav2)
    conf = (conf - np.min(conf)) / (np.max(conf) - np.min(conf))
    confidence = [np.max(conf[x-conf_interval:x+conf_interval]) for x in onsets2]
    onsets2 = onset_nms(onsets2, confidence)
    plt.plot(np.arange(len(wav2)), wav2)
    if ap != None:
        plt.text(44000, np.max(wav2) + 0.2, f"AP: {ap:.2f}")
    for idx, x in enumerate(onsets2):
        confidence = np.max(conf[x-conf_interval:x+conf_interval])
        plt.text(x+10, np.min(wav2) - 0.2, f"conf: {confidence:.2f}")
        if pred != None:
            if pred[idx]:
                plt.vlines(x, np.min(wav2) - 0.2, np.max(wav2) + 0.2, colors='green')
            else:
                plt.vlines(x, np.min(wav2) - 0.2, np.max(wav2) + 0.2, colors='r')
        else:
            plt.vlines(x, np.min(wav2) - 0.2, np.max(wav2) + 0.2, colors='black')
    plt.savefig(path)
    plt.cla()
    plt.close()

if __name__ == '__main__':
    if args.longer_det:
        duration = int(args.gen_dir.split('_')[-2].replace('s', ''))
        key_map = json.load(open('data/AMT_test_8s/key_map.json', 'r'))
        tar_npy_path = args.tar_dir + f'_{duration}s_onsets_wav.npy'
    else:
        tar_npy_path = args.tar_dir + f'_onsets_wav.npy'
        duration = 2
    gen_npy_path = args.gen_dir + '_onsets_wav.npy'
    # if os.path.exists(tar_npy_path):
    #     print(tar_npy_path)
    #     onset_list1, wav_list1 = np.load(tar_npy_path, allow_pickle=True)
    # else:
    onset_list1, wav_list1 = detect_onset(args.tar_dir, duration=duration)
    np.save(tar_npy_path, (onset_list1, wav_list1))
    # if os.path.exists(gen_npy_path):
    #     onset_list2, wav_list2 = np.load(gen_npy_path, allow_pickle=True)
    # else:
    onset_list2, wav_list2 = detect_onset(args.gen_dir, duration=duration)
    np.save(gen_npy_path, (onset_list2, wav_list2))
    if args.plt:
        plot_dir = os.path.join('/homes/emilpost/riccardo/diffusion-sfx/onset_detection_vis_completed_aug/', args.gen_dir.split('/')[-1])
        os.makedirs(plot_dir, exist_ok=True)
    cnt_match = []
    onset_acc = []
    onset_ap = []

    for generated_audio_name, onset2 in onset_list2.items():
        tar_key = os.path.splitext(generated_audio_name)[0] + '.wav'

        onset1 = onset_list1.get(tar_key, []) 
        
        if len(onset1) == 0 or len(onset2) == 0:
            
            cnt_match.append(False) 
            onset_acc.append(0.0)  
            onset_ap.append(0.0)  
        else:
            if args.remove_head is not None:
                onset1 = np.array([o1 for o1 in onset1 if o1 >= args.remove_head * 22050])
                onset2 = np.array([o2 for o2 in onset2 if o2 >= args.remove_head * 22050])
            cnt_match.append(len(onset1) == len(onset2))
            if args.multi_delta:
                delta_list = list(np.arange(0.1, args.delta + 0.05, 0.05))
                acc = 0
                ap = 0
                for delta in delta_list:
                    _acc, _ap, onset2_res = eval_osnets(onset1, onset2, wav_list2[generated_audio_name], delta=delta, keys=(tar_key, generated_audio_name))
                    acc += _acc
                    ap += _ap
                acc /= len(delta_list)
                ap /= len(delta_list)
            else:
                acc, ap, onset2_res = eval_osnets(onset1, onset2, wav_list2[generated_audio_name], delta=args.delta, keys=(tar_key, generated_audio_name))
            if args.plt:
                plot_onset(os.path.join(plot_dir, f'onset_wav_tar_{tar_key}_cond_{generated_audio_name}.jpg'), wav_list1[tar_key], wav_list2[generated_audio_name], onset1, onset2, pred=onset2_res, ap=ap)
            onset_acc.append(acc)
            onset_ap.append(ap)

    print(f'#onset acc: {np.mean(cnt_match):.4f}, detection acc: {np.mean(onset_acc):.4f}, detection ap: {np.mean(onset_ap):.4f}')
