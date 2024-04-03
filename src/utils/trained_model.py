from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np


@dataclass
class TrainedModel:
    w: np.ndarray
    b: float


def read_model(path: Path) -> TrainedModel:
    with path.open("rb") as source:
        return pickle.load(source)


def write_model(model: TrainedModel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as dest:
        pickle.dump(model, dest)
