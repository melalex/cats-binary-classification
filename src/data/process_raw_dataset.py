from pathlib import Path
from PIL import Image

import random
import zipfile
import logging
import logging.config

from definitions import (
    CAT_SAMPLE_RATIO,
    CATS_DATASET_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOG_PERIOD,
    LOGGING_CONFIG_PATH,
    RAW_DATA_FOLDER,
    SAMPLE_DATASET_NAME,
    TEST_DATA_FOLDER,
    TEST_TRAIN_RATIO,
    TRAIN_DATA_FOLDER,
    VALID_DATASET_FLAG_FILE,
)


def process_raw_dataset(logger: logging.Logger):
    if VALID_DATASET_FLAG_FILE.is_file():
        logger.info("Dataset is already prepared. Skipping ...")
        return

    cats_dataset_path = unzip_dataset(CATS_DATASET_NAME, logger) / "PetImages" / "Cat"
    sample_dataset_path = unzip_dataset(SAMPLE_DATASET_NAME, logger) / "data"
    cat_images, sample_images = list_images(
        cats_dataset_path, sample_dataset_path, logger
    )
    train_cat_images_count = int(len(cat_images) * TEST_TRAIN_RATIO)
    train_sample_images_count = int(len(sample_images) * TEST_TRAIN_RATIO)

    logger.info(
        "Preparing [ %s ] cat images and [ %s ] for train dataset",
        train_cat_images_count,
        train_sample_images_count,
    )
    logger.info(
        "Preparing [ %s ] cat images and [ %s ] for test dataset",
        len(cat_images) - train_cat_images_count,
        len(sample_images) - train_sample_images_count,
    )

    train_cat_images = cat_images[:train_cat_images_count]
    test_cat_images = cat_images[train_cat_images_count:]
    train_sample_images = sample_images[:train_sample_images_count]
    test_sample_images = sample_images[train_sample_images_count:]

    train_dataset = train_cat_images + train_sample_images
    test_dataset = test_cat_images + test_sample_images

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    write_to(train_dataset, TRAIN_DATA_FOLDER, logger)
    write_to(test_dataset, TEST_DATA_FOLDER, logger)

    VALID_DATASET_FLAG_FILE.touch()


def unzip_dataset(name: str, logger: logging.Logger) -> Path:
    zip_name = name + ".zip"
    extract_to = RAW_DATA_FOLDER / name

    if extract_to.is_dir():
        logger.info("[ %s ] is already unzipped. Skipping ...", name)
    else:
        logger.info("Unzipping [ %s ] to [ %s ]", name, extract_to)
        with zipfile.ZipFile(RAW_DATA_FOLDER / zip_name, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_FOLDER / name)

    return extract_to


def list_images(
    cats_dataset_path: Path, sample_dataset_path: Path, logger: logging.Logger
) -> tuple[list[Path], list[Path]]:
    cat_images = [it for it in cats_dataset_path.iterdir() if it.suffix == ".jpg"]
    sample_images = [it for it in sample_dataset_path.iterdir() if it.suffix == ".jpg"]

    if len(cat_images) < len(sample_images):
        cat_images_count = len(cat_images)
        sample_images_count = CAT_SAMPLE_RATIO * cat_images_count
    else:
        sample_images_count = len(sample_images)
        cat_images_count = sample_images_count // CAT_SAMPLE_RATIO

    logger.info(
        "Found [ %s ] cat and [ %s ] sample images",
        cat_images_count,
        sample_images_count,
    )

    return (
        cat_images[:cat_images_count],
        sample_images[:sample_images_count],
    )


def write_to(source: list[Path], path: Path, logger: logging.Logger):
    path.mkdir(parents=True, exist_ok=True)

    img_count = len(source)
    dig_count = len(str(img_count))

    for i in range(img_count):
        it = source[i]
        img_type = "Y" if CATS_DATASET_NAME in str(it) else "N"
        new_file_name = path / f"{str(i).zfill(dig_count)}-{img_type}.jpg"
        img = Image.open(it)
        resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        resized.convert("RGB").save(new_file_name)
        if i % LOG_PERIOD == 0:
            logger.info("Prepared [ %s ] of [ %s ] images", i, img_count)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)
    process_raw_dataset(logging.getLogger(__name__))
