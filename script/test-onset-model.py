import torch
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision("high")


def cli_main():
    # run = wandb.init(
    #     entity="team-diffusion-sfx",
    #     project="diffusion-sfx",
    #     id=None, # if null wandb gives it id
    #     name="GH-test", # if null wandb gives it name
    #     dir="logs/onset/test", # dir needs to already exist
    #     group="ONSET-TEST"
    # )

    cli = LightningCLI(
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "ddp",  # DDPStrategy(find_unused_parameters=False),
            "devices": -1,
            "num_sanity_val_steps": 2,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 100,
            "sync_batchnorm": True,
            "enable_model_summary": True,
            "benchmark": False,
            "enable_checkpointing": True,
        },
        save_config_kwargs={
            # "config_filename": "config.yaml",
            "overwrite": True,
        }
    )

    # run.finish()


if __name__ == "__main__":
    cli_main()
