#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: predict_segmentation.py
Author: nicolas-heigvd
Date: 2025-09-12
Version: 1.0
Description: This script is an enhancement from the R script predict_segmentation.py
"""

# %%
import sys
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from logging_config import logger


# %%
def run_inference(model_path: Path, data_path: Path):
    """Docstring"""
    start = time()
    logger.info(
        f"Starting deep learning inference using model={model_path.as_posix()} and data_path={data_path.as_posix()}"
    )
    # Set paths
    PICS_PATH = Path(data_path / "pics")
    MASKS_PATH = Path(data_path / "masks")
    if not MASKS_PATH.is_dir():
        MASKS_PATH.mkdir(exist_ok=True)

    # Define image sizes
    img_size_org = (512, 512)
    n_splits = 2
    img_size = (
        np.rint(img_size_org[0] / n_splits).astype(int),
        np.rint(img_size_org[1] / n_splits).astype(int),
    )

    # Load model
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "loss": None,
            "falsepos": None,
            "falseneg": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
        },
    )

    included_extensions = ["png"]
    files = [
        f.name
        for f in PICS_PATH.iterdir()
        if f.is_file() and f.suffix.lower().lstrip(".") in included_extensions
    ]
    for file in files:
        try:
            file_path = Path(PICS_PATH / file)
            image = tf.io.read_file(file_path.as_posix())
            image = tf.image.decode_jpeg(image)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, img_size_org)
            patches_img = tf.image.extract_patches(
                tf.reshape(image, [1, img_size_org[0], img_size_org[1], 1]),
                [1, img_size[0], img_size[1], 1],
                [1, img_size[0], img_size[1], 1],
                [1, 1, 1, 1],
                padding="VALID",
            )
            patches_img = tf.reshape(
                patches_img, [n_splits**2, img_size[0], img_size[1], 1]
            )
            pred = model.predict(patches_img)

            # np.unique(np.round(pred[2,:,:,0],2), return_counts=True)

            for i in range(n_splits * n_splits):
                img = keras.preprocessing.image.array_to_img(
                    np.expand_dims(pred[i, :, :, 1], -1)
                )
                img = img.resize((150, 150))
                output_path = Path(MASKS_PATH) / f"{Path(file).stem}_{i}.png"
                img.save(output_path)

        except Exception as err:
            logger.error(f"Error with file: {file}\n{err}")

    delta_t = time() - start
    logger.info(f"Inference took {delta_t:.2f} seconds.")


def main():
    """Main function"""
    if len(sys.argv) != 3:
        logger.info("Usage: predict_segmentation.py <model_path> <data_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    data_path = Path(sys.argv[2])

    run_inference(model_path, data_path)
    logger.info(f'Script "{__file__}" run successfully!')


# Execute prediciton for all images in folder
if __name__ == "__main__":
    main()
