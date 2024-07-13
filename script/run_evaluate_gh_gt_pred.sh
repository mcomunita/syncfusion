# creates ground truth data for the diffusion + predicted onsets experiment
HYDRA_FULL_ERROR=1 \
PYTHONPATH=. \
python evaluate.py \
exp=evaluate_gh_gt_pred \
experiment.dataset.path="/import/c4dm-datasets-ext/DIFF-SFX-webdataset/greatest_hits/test_onset_preds.tar"
