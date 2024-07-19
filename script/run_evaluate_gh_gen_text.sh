# evaluates FAD for diffusion only conditioning with text
# comment last 3 lines if you already have the generated data
python script/evaluate_diffusion.py \
exp=evaluate_gh_gen_text \
experiment_path=./output/experiments/gh-gen-text \
evaluation.gt_path=./output/experiments/gh-gt \
experiment.dataset.path="data/greatest-hits/webdataset/test_shard_1.tar" \
model.embedder_checkpoint="checkpoints/630k-audioset-best.pt" \
experiment.model_path="checkpoints/epoch\=784-valid_loss\=0.008.ckpt"