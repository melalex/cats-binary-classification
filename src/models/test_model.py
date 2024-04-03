import logging
import logging.config
from pathlib import Path

import numpy as np

from definitions import (
    LOGGING_CONFIG_PATH,
    MODEL_PATH,
    TEST_DATA_FOLDER,
    TRAIN_DATA_FOLDER,
)
from models.predict import predict_with_model
from utils.image import extract_prediction, read_all_images_from
from utils.trained_model import TrainedModel, read_model


def test_model(logger: logging.Logger):
    model = read_model(MODEL_PATH)
    train_result = test_model_with(model, TRAIN_DATA_FOLDER)
    test_result = test_model_with(model, TEST_DATA_FOLDER)

    logger.info("Train accuracy: %s %%", train_result)
    logger.info("Test accuracy: %s %%", test_result)


def test_model_with(model: TrainedModel, path: Path) -> float:
    predict_all = np.vectorize(predict_with_model)
    extract_all_prediction = np.vectorize(extract_prediction)
    imgs = read_all_images_from(path)
    a = predict_all(imgs, model)
    y = extract_all_prediction(imgs)

    return 100 - np.mean(np.abs(a - y)) * 100


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    test_model(logging.getLogger(__name__))
