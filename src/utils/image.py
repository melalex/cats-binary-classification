from dataclasses import dataclass
from pathlib import Path
from PIL import Image

import numpy as np

from definitions import IMAGE_HEIGHT, IMAGE_WIDTH


def prepare_image(path: Path):
    img = Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    return normalize_image(np.array(img))


def normalize_image(x: np.ndarray) -> np.ndarray:
    return x / 255


def extract_prediction(path: Path) -> int:
    return 1 if "Y" in path.stem else 0


def read_all_images_from(path: Path) -> list[Path]:
    imgs = [it for it in path.iterdir()]

    imgs.sort()

    return imgs
