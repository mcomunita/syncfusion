# creates ground truth data for the diffusion only experiments
HYDRA_FULL_ERROR=1 \
PYTHONPATH=. \
python script/evaluate.py \
exp=evaluate_gh_gt \
experiment.dataset.path="/import/c4dm-datasets-ext/DIFF-SFX-webdataset/greatest_hits/test_shard_1.tar"