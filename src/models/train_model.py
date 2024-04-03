from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import logging
import logging.config

from definitions import (
    ITERATION_COUNT,
    LEARNING_RATE,
    LOG_PERIOD,
    LOGGING_CONFIG_PATH,
    MODEL_PATH,
    TRAIN_DATA_FOLDER,
)
from utils.fun import logistic_regression
from utils.image import extract_prediction, prepare_image, read_all_images_from
from utils.trained_model import TrainedModel, write_model


def train_model(
    dataset: Path, learning_factor: float, iter_count: int, logger: logging.Logger
):
    if MODEL_PATH.is_file():
        logger.info(
            "Model file is present at [ %s ]. Skipping training ...", MODEL_PATH
        )
        return

    imgs = read_all_images_from(dataset)

    x = adjust_shape(np.stack([prepare_image(it) for it in imgs]))
    y = np.array([extract_prediction(it) for it in imgs]).reshape((1, -1))

    model = gradient_descent(x, y, learning_factor, iter_count, logger)

    logger.info("Saving trained model to [ %s ]", MODEL_PATH)

    write_model(model, MODEL_PATH)


def adjust_shape(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).T


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    iter_count: int,
    logger: logging.Logger,
) -> TrainedModel:
    m = x.shape[1]
    w = np.zeros((x.shape[0], 1))
    b = 0.0

    logger.info("Found [ %s ] training images", m)

    for i in range(iter_count):
        a = logistic_regression(x, w, b)

        if i % LOG_PERIOD == 0 and logger.isEnabledFor(logging.INFO):
            logger.info(
                "Iteration # [ %s ] of [ %s ]. Cost is [ %s ]",
                i,
                iter_count,
                calculate_cost(y, a),
            )

        dz = a - y
        dw = (1 / m) * np.dot(x, dz.T)
        db = (1 / m) * np.sum(dz)
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return TrainedModel(w, b)


def calculate_cost(y: np.ndarray, a: np.ndarray) -> float:
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    train_model(
        TRAIN_DATA_FOLDER, LEARNING_RATE, ITERATION_COUNT, logging.getLogger(__name__)
    )
