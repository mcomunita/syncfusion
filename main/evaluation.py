from pathlib import Path
from typing import Union

import pandas as pd
from frechet_audio_distance import FrechetAudioDistance

def evaluate_fad(
        experiment_path: Union[str, Path],
        gt_path: Union[str, Path]
) -> pd.DataFrame:
    generation_path = Path(experiment_path)
    original_path = Path(gt_path)

    assert generation_path.exists(), generation_path
    assert original_path.exists(), original_path

    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False,
        use_activation=False,
        verbose=True
    )

    fad_score = frechet.score(original_path, generation_path, dtype="float32")
    df = pd.DataFrame(columns=["FAD"])
    df.loc[0] = fad_score
    print(f"FAD = {fad_score}")
    return df