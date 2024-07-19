import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import wandb
from pytorch_lightning.cli import LightningCLI


torch.set_float32_matmul_precision("high")


def cli_main():
    _ = LightningCLI()


if __name__ == "__main__":
    cli_main()
    wandb.finish()