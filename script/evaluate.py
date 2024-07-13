from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from main import utils

dotenv.load_dotenv(override=True)
log = utils.get_logger(__name__)


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    experiment_path = Path(config.experiment_path)
    pl.seed_everything(config.seed)

    # Experiment dataset
    if "experiment" in config:
        experiment_fn = hydra.utils.instantiate(config.experiment)
        experiment_fn()

    # Compute metrics
    if "evaluation" in config:
        evaluate_fn = hydra.utils.instantiate(config.evaluation)
        results = evaluate_fn()

        # Store and show results
        results.to_csv(experiment_path / "metrics.csv")


if __name__ == "__main__":
    main()
