python script/train_diffusion_model.py \
exp=train_diffusion_gh \
model.embedder_checkpoint="checkpoints/630k-audioset-best.pt" \
datamodule.batch_size=2 \
datamodule.num_workers=8 \
datamodule.train_dataset.path="data/greatest-hits/webdataset/train_shard_\{1..3\}.tar" \
datamodule.val_dataset.path="data/greatest-hits/webdataset/val_shard_1.tar"