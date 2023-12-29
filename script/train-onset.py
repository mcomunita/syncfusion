import os
import torch
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy

import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

torch.set_float32_matmul_precision("high")


def cli_main():
    # wandb.init(
    #     entity="team-mcomunita",
    #     project="diffusion-sfx",
    #     id=None,  # if null wandb gives it id
    #     name="GH-train",  # if null wandb gives it name
    #     dir="/import/c4dm-datasets-ext/DIFF-SFX/logs/onset-temp/train",  # dir needs to already exist
    #     group="ONSET-TRAIN"
    # )

    cli = LightningCLI(
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": DDPStrategy(find_unused_parameters=False),
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
            "overwrite": True,
        }
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()

    # local_rank = torch.distributed.get_rank()

    # print("local_rank", local_rank)

    # print("args.local_rank", args.local_rank)

    cli_main()
