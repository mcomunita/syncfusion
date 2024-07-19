import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import wandb
from pytorch_lightning.cli import LightningCLI
# from pytorch_lightning.strategies import DDPStrategy


torch.set_float32_matmul_precision("high")


def cli_main():
    _ = LightningCLI(
        trainer_defaults={
            "accelerator": "gpu",
            # "strategy": "ddp",
            # "strategy": DDPStrategy(find_unused_parameters=False),
            "devices": -1,
            "num_sanity_val_steps": 2,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 10,
            "sync_batchnorm": True,
            "enable_model_summary": True,
            "benchmark": True,
            "enable_checkpointing": True,
        },
        save_config_kwargs={
            # "config_filename": "config.yaml",
            "overwrite": False,
        }
    )


if __name__ == "__main__":
    cli_main()
    wandb.finish()