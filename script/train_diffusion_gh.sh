CUDA_VISIBLE_DEVICES=1 \
python script/train-diffusion.py exp=train_diffusion_gh \
model.embedder_checkpoint=checkpoints/630k-audioset-best.pt \
datamodule.batch_size=2 \
datamodule.train_dataset.path="data/DIFF-SFX-webdataset/greatest_hits/train_shard_\{1..3\}.tar" \
datamodule.val_dataset.path="data/DIFF-SFX-webdataset/greatest_hits/val_shard_1.tar"