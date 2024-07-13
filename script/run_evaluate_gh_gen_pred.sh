# evaluates FAD for diffusion + predicted onsets without augmentation
HYDRA_FULL_ERROR=1 \
PYTHONPATH=. \
python evaluate.py \
exp=evaluate_gh_gen_pred \
experiment.dataset.path="/import/c4dm-datasets-ext/DIFF-SFX-webdataset/greatest_hits/test_onset_preds.tar" \
model.embedder_checkpoint="/homes/emilpost/diffusion-sfx/data/630k-audioset-best.pt" \
experiment.model_path="/homes/emilpost/diffusion-sfx/logs/ckpts/2023-08-24-12-23-39/epoch\=784-valid_loss\=0.008.ckpt" \
experiment_path=./output/experiments/gh-gen-pred \
evaluation.gt_path=./output/experiments/gh-gt-pred