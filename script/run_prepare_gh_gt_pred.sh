# creates ground truth data for the diffusion + predicted onsets experiment
python script/evaluate_diffusion.py \
exp=prepare_gh_gt_pred \
experiment.dataset.path="data/greatest-hits/webdataset/test_onset_preds.tar"
