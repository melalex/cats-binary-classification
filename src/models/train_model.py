from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import numpy as np
import logging
import logging.config
import pickle

from definitions import (
    ITERATION_COUNT,
    LEARNING_RATE,
    LOGGING_CONFIG_PATH,
    MODELS_FOLDER,
    TRAIN_DATA_FOLDER,
)


@dataclass
class TrainedModel:
    w: np.ndarray
    b: float


def train_model(
    dataset: Path, learning_factor: float, iter_count: int, logger: logging.Logger
):
    imgs = [it for it in dataset.iterdir()]
    x = np.array([read_image(it) for it in imgs])
    y = np.array([1 if "Y" in it.stem else 0 for it in imgs])

    logger.info("Found [ %s ] training images", len(x))

    model = gradient_descent(x, y, learning_factor, iter_count, logger)
    model_path = (
        MODELS_FOLDER / f"cats-binary-classification-{learning_factor}-{iter_count}.bin"
    )

    logger.info("Saving trained model to [ %s ]", model_path)

    with model_path.open("wb") as dest:
        pickle.dump(model, dest)


def read_image(it: Path) -> np.ndarray:
    img = Image.open(it)
    x = np.array(img)

    return normalize_image(x.reshape(x.shape[0], -1).T)


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    learning_factor: float,
    iter_count: int,
    logger: logging.Logger,
) -> TrainedModel:
    m = x.shape[1]
    w = np.zeros((x.shape[0], 1))
    b = 0.0

    for i in range(iter_count):
        z = logistic_regression(x, w, b)
        a = sigmoid(z)
        dz = a - y
        dw = (1 / m) * x * dz.T
        db = (1 / m) * np.sum(dz)
        w = w - learning_factor * dw
        b = b - learning_factor * db
        if i % 100 == 0 and logger.isEnabledFor(logging.INFO):
            logger.info("Iteration # [ %s ]. Cost is [ %s ]", i, calculate_cost(y, a))

    return TrainedModel(w, b)


def calculate_cost(y: np.ndarray, y_predicted: np.ndarray) -> float:
    return np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))


def logistic_regression(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.dot(w.T, x) + b


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def normalize_image(x: np.ndarray) -> np.ndarray:
    return x / 255


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    train_model(
        TRAIN_DATA_FOLDER, LEARNING_RATE, ITERATION_COUNT, logging.getLogger(__name__)
    )
