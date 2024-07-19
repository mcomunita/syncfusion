# evaluates FAD for diffusion + predicted onsets with augmentation
# comment last 3 lines if you already have the generated data
# and comment the experiment in the yaml file
python script/evaluate_diffusion.py \
exp=evaluate_gh_gen_pred_augment \
experiment_path=./output/experiments/gh-gen-pred-augment \
evaluation.gt_path=./output/experiments/gh-gt-pred \
experiment.dataset.path="data/greatest-hits/webdataset/test_onset_augment_preds.tar" \
model.embedder_checkpoint="checkpoints/630k-audioset-best.pt" \
experiment.model_path="checkpoints/epoch\=784-valid_loss\=0.008.ckpt"