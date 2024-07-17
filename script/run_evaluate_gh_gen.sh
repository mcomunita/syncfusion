# evaluates FAD for diffusion only conditioning with audio
python script/evaluate.py \
exp=evaluate_gh_gen \
experiment.dataset.path="data/DIFF-SFX-webdataset/greatest_hits/test_shard_1.tar" \
model.embedder_checkpoint="checkpoints/630k-audioset-best.pt" \
experiment.model_path="checkpoints/epoch\=784-valid_loss\=0.008.ckpt" \
experiment_path=./output/experiments/gh-gen \
evaluation.gt_path=./output/experiments/gh-gt