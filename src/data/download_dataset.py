import kaggle
import logging
import logging.config

from definitions import (
    CATS_DATASET_NAME,
    CATS_DATASET_OWNER,
    PROJECT_ROOT_DIR,
    RAW_DATA_FOLDER,
    SAMPLE_DATASET_NAME,
    SAMPLE_DATASET_OWNER,
)


def download_dataset(owner: str, name: str, logger: logging.Logger):
    file_path = RAW_DATA_FOLDER / f"{name}.zip"

    if file_path.is_file():
        logger.info(
            "Found [ %s ] dataset in [ %s ]. Skipping download...", name, file_path
        )
    else:
        logger.info("Downloading [ %s ] dataset to [ %s ]", name, file_path)
        kaggle.api.dataset_download_files(
            dataset=f"{owner}/{name}", path=RAW_DATA_FOLDER
        )


def download_all_datasets(logger: logging.Logger):
    kaggle.api.authenticate()
    download_dataset(CATS_DATASET_OWNER, CATS_DATASET_NAME, logger)
    download_dataset(SAMPLE_DATASET_OWNER, SAMPLE_DATASET_NAME, logger)


if __name__ == "__main__":
    logging.config.fileConfig(PROJECT_ROOT_DIR / "logging.ini")
    download_all_datasets(logging.getLogger(__name__))
