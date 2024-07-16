# evaluates FAD for diffusion only conditioning with text
python script/evaluate.py \
exp=evaluate_gh_gen_text \
experiment.dataset.path="data/DIFF-SFX-webdataset/greatest_hits/test_shard_1.tar" \
model.embedder_checkpoint="checkpoints/630k-audioset-best.pt" \
experiment.model_path="/homes/emilpost/diffusion-sfx/logs/ckpts/2023-08-24-12-23-39/epoch\=784-valid_loss\=0.008.ckpt" \
experiment_path=./output/experiments/gh-gen-text \
evaluation.gt_path=./output/experiments/gh-gt
