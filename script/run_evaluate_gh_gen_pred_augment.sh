# evaluates FAD for diffusion + predicted onsets with augmentation
python script/evaluate.py \
exp=evaluate_gh_gen_pred \
experiment.dataset.path="data/DIFF-SFX-webdataset/greatest_hits/test_onset_augment_preds.tar" \
model.embedder_checkpoint="checkpoints/630k-audioset-best.pt" \
experiment.model_path="/homes/emilpost/diffusion-sfx/logs/ckpts/2023-08-24-12-23-39/epoch\=784-valid_loss\=0.008.ckpt" \
experiment_path=./output/experiments/gh-gen-pred-augment \
evaluation.gt_path=./output/experiments/gh-gt-pred