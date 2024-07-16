# creates ground truth data for the diffusion + predicted onsets experiment
python script/evaluate.py \
exp=evaluate_gh_gt_pred \
experiment.dataset.path="data/DIFF-SFX-webdataset/greatest_hits/test_onset_preds.tar"
