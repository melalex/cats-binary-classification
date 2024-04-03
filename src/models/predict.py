from pathlib import Path
import pickle

from definitions import IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH
from utils.fun import logistic_regression
from utils.image import prepare_image
from utils.trained_model import TrainedModel, read_model


def predict(path: Path) -> int:
    return predict_with_model(path, read_model(MODEL_PATH))


def predict_with_model(path: Path, model: TrainedModel) -> int:
    img = prepare_image(path).reshape((1, IMAGE_WIDTH * IMAGE_HEIGHT * 3)).T

    prediction = logistic_regression(img, model.w, model.b)

    if prediction > 0.5:
        return 1
    else:
        return 0
