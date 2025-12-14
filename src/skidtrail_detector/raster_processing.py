#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: raster_preprocessing.py
Author: nicolas-heigvd
Date: 2025-10-10
Version: 1.0
Description: This script is a conversion from the R script raster_preprocessing.R
"""

# %%
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import array_bounds, from_origin

from config import config

# from error_handling import check_file
from logging_config import logger

# %%


# Functions
def preprocess_raster(
    ground_structure_path, fd, target_path, w, window_size, grid_id_path
):
    """Docstring"""
    path_wa_folder = Path(target_path) / str(w)
    path_wa_folder.mkdir(parents=True, exist_ok=True)


def postprocess_raster(
    i: int,
    target_path: Path,
    forest_raster_mask_path: Path,
    threshold_segmentation: float,
):
    """Post process raster tiles from tf output"""
    target_path = Path(target_path)
    result_path = target_path / f"segmentation_results_{i}.tif"
    path_masks = target_path / "masks"

    with rasterio.open(forest_raster_mask_path) as src:
        forest_raster_mask = src.read(1)
        original_width = src.width
        original_height = src.height
        mask_meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        origin = (bounds.left, bounds.top)

    tile_height = tile_width = 2 * config.WINDOW_SIZE
    nrows, ncols = forest_raster_mask.shape
    final_height = nrows * tile_height
    final_width = ncols * tile_width
    img_tot = np.zeros((final_height, final_width), dtype=np.uint8)

    for r in range(nrows):
        for c in range(ncols):
            cell_id = forest_raster_mask[r, c]
            y0 = r * tile_height
            x0 = c * tile_width
            # logger.info(f"Computing tile at ({x0=},{y0=})")
            # TODO: Fix the 150m/300px shift to the right...
            # it may be related to the tiling scheme which can already be shifted...
            if np.isnan(cell_id) or int(cell_id) == 0 or int(cell_id) == -9999:
                tile = np.zeros((tile_height, tile_width), dtype=np.uint8)
            else:
                try:
                    parts = []
                    for idx in range(4):
                        part_path = path_masks / f"tile_{int(cell_id)}_{idx}.png"
                        part = Image.open(part_path).convert("L")
                        parts.append(np.array(part))

                    # Assemble 2x2 tile: [0 1]
                    #                    [2 3]
                    top = np.hstack((parts[0], parts[1]))
                    bottom = np.hstack((parts[2], parts[3]))
                    tile = np.vstack((top, bottom))
                except Exception as err:
                    logger.info(f"Failed to load tile {cell_id}: {err}")
                    tile = np.zeros((tile_height, tile_width), dtype=np.uint8)

            img_tot[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile

    # Normalize to [0, 1] range
    img_normalized = img_tot.astype(np.float32) / 255.0
    # Apply threshold
    img_binary = np.where(img_normalized >= threshold_segmentation, 1, 0).astype(
        np.uint8
    )

    # Convert to raster
    new_width = img_binary.shape[1]  # cols
    new_height = img_binary.shape[0]  # rows
    new_pixel_width = (transform.a * original_width) / new_width
    new_pixel_height = (-transform.e * original_height) / new_height
    new_transform = from_origin(origin[0], origin[1], new_pixel_width, new_pixel_height)

    mask_meta.update(
        {
            # "dtype": "uint8",
            "count": 1,
            "transform": new_transform,
            "height": final_height,
            "width": final_width,
            "crs": crs,
            "compress": "lzw",
        }
    )

    # Save raster
    with rasterio.open(result_path, "w", **mask_meta) as dst:
        dst.write(img_binary, 1)

    logger.info(f"Segmentation outputs joined on forest part {i}: {result_path}")


def get_extent_from_profile(profile):
    """
    Compute raster extent from a rasterio profile.
    Returns (minx, miny, maxx, maxy).
    """
    h = profile["height"]
    w = profile["width"]
    transform = profile["transform"]

    return array_bounds(h, w, transform)


def build_vrt_file(input_dem_dir: Path) -> None:
    """Build VRT file from a list of DEM filenames"""
    logger.info(f"Building VRT file for {input_dem_dir=}")
    url = "http://gdal:5001/build_vrt"
    if not input_dem_dir:
        logger.warning(
            f"File {input_dem_dir} does not exist or is not defined, setting it to defaulf {config.INPUT_DEM_DIR}."
        )
        input_dem_dir = config.INPUT_DEM_DIR

    input_pattern = input_dem_dir / "*.tif"
    output = config.TEMP_DEM_DIR / "swissalti3d_Vaud_0.5_2056_5728.vrt"
    if "diff" in input_dem_dir.as_posix().lower():
        output = config.TEMP_DEM_DIR / "swissalti3d_Vaud_0.5_2056_5728_diff.vrt"

    payload = {
        "input_pattern": input_pattern.as_posix(),
        "output": output.as_posix(),
    }
    response = requests.post(url, json=payload, timeout=(5, 30))
    if response.status_code != 200:
        response.raise_for_status()
    else:
        logger.info(f"VRT file written to {output} successfully!")
