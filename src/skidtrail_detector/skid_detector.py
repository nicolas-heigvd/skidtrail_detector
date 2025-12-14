#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: main.py
Author: nicolas-heigvd
Date: 2025-09-12
Version: 1.0
Description: This script is a conversion from the R script main.R
"""

# %%
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import time

import fiona
import geopandas as gpd
import numpy as np
import rasterio as rio
from PIL import Image
from rasterio.mask import mask
from rasterio.transform import from_origin, xy
from rasterio.windows import from_bounds

from config import config
from logging_config import logger
from skidtrail_detector.fetch_swissalti import (
    crop_gdf_to_available_dem_extent,
)
from skidtrail_detector.line_finder import line_finder, line_finder_skel
from skidtrail_detector.predict_segmentation import run_inference
from skidtrail_detector.raster_processing import postprocess_raster


# %%
def build_grid(gdf):
    """Docstring"""
    logger.info(f"{config.DELINEATION_DIR=}")
    # minx, miny, maxx, maxy = gdf.total_bounds
    bounds = np.array(gdf.total_bounds)
    expand = 200
    expand_arr = np.array([-expand, -expand, expand, expand]) / 2
    expanded_bounds = bounds + expand_arr

    minx, miny, maxx, maxy = expanded_bounds

    width = int(np.round((maxx - minx) / config.WINDOW_SIZE))
    height = int(np.round((maxy - miny) / config.WINDOW_SIZE))
    transform = from_origin(minx, maxy, config.WINDOW_SIZE, config.WINDOW_SIZE)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int32",
        "crs": gdf.crs.to_string(),
        "transform": transform,
    }

    grid_id_array = np.arange(1, height * width + 1, dtype=np.int32).reshape(
        (height, width)
    )

    # Write raster to file, overwrite if exists
    with rio.open(config.PREDICTION_FILE, "w", **profile) as dst:
        dst.write(grid_id_array, 1)

    logger.info(
        f"Raster saved to file: {config.PREDICTION_FILE}"
    )  # corresponds to main.R:l56


def extract_tiles(i, forest_geom):
    """Build the raster image mask covering the given forest geometry.
    Each pixel of the raster image mask is a 150x150m tile with a resolution
    corresponding to the input DEM file (usually 0.5m/px)."""
    # early return
    if forest_geom.area < config.MIN_FOREST_AREA:
        return
    # logger.info(f"Processing feature {i}...")
    CROP_DIR = Path(config.PREDICTION_DIR / f"{i}")
    CROP_DIR.mkdir(exist_ok=True)

    USE_BUFFER = True
    if USE_BUFFER:
        forest_geom = forest_geom.buffer(np.sqrt(2 * (config.WINDOW_SIZE**2)) / 2)
        SAVE_BUFFER = True
        if SAVE_BUFFER:
            schema = {
                "geometry": "Polygon",
                "properties": {"id": "int"},
            }
            BUFFER_FILE_PATH = Path(CROP_DIR / f"buffer_{i}.shp")
            with fiona.open(
                BUFFER_FILE_PATH,
                mode="w",
                driver="ESRI Shapefile",
                schema=schema,
                crs="EPSG:2056",
            ) as layer:
                layer.write(
                    {
                        "geometry": forest_geom,
                        "properties": {"id": 1},
                    }
                )

    # Open DEM_FILE to get some metadata only:
    with rio.open(config.DEM_FILE) as src:
        input_nodata = src.nodata
        logger.info(f"DEM {input_nodata=}")
        input_crs = src.crs
        # input_dtype = src.dtypes[0]  # TODO: check

    with rio.open(config.PREDICTION_FILE) as grid_src:  #  beware: integer dtype!
        geom_crop, wa_transform = mask(
            grid_src,
            [forest_geom],
            all_touched=True,
            nodata=input_nodata,  # <- check that this matches the input_dtype
            crop=True,
        )

    # corresponds to main.R:l64
    FOREST_RASTER_MASK_PATH = Path(CROP_DIR / f"forest_raster_mask_{i}.tif")
    with rio.open(
        FOREST_RASTER_MASK_PATH,
        "w",
        driver="GTiff",
        height=geom_crop.shape[1],  # rows
        width=geom_crop.shape[2],  # cols
        count=geom_crop.shape[0],  # bands (usually 1)
        dtype=geom_crop.dtype,
        crs=input_crs,
        transform=wa_transform,
        nodata=input_nodata,
    ) as dst:
        dst.write(geom_crop)

    return (geom_crop, wa_transform)


def get_extent_of_array_with_transform(array, transform):
    """Get the extent of an array given an affine transform"""
    _, height, width = array.shape
    xmin, ymax = transform * (0, 0)  # top-left corner of the raster
    xmax, ymin = transform * (width, height)  # bottom-right corner of the raster

    # Bounding box (xmin, ymin, xmax, ymax)
    return (xmin, ymin, xmax, ymax)


def build_tile(i, geom_crop, wa_transform, src):
    """Build the tiled PNG images (300x300px) over a single forest
    in order to run the inference.
    """
    CROP_DIR = Path(config.PREDICTION_DIR / f"{i}")
    PICS_DIR = Path(CROP_DIR / "pics")
    PICS_DIR.mkdir(exist_ok=True)
    _, cols = geom_crop.shape[1:]
    flat_geom_crop = geom_crop[0].flatten()
    logger.info(
        f"Processing raster grids for forest {i}, please wait, this may take a while..."
    )
    for idx, id_cell in enumerate(flat_geom_crop):
        if id_cell == 0 or np.isnan(id_cell) or int(id_cell) == -9999:
            continue
        PIC_PATH = Path(PICS_DIR / f"tile_{id_cell}.png")
        r = (idx) // cols
        c = (idx) % cols
        logger.debug(f"{idx=},{id_cell=},row={r},col={c}")
        x, y = xy(wa_transform, r, c)
        offsets = np.array([-1, -1, 1, 1]) * config.HALF_WINDOW
        cell_ext = np.array([x, y, x, y]) + offsets

        # Windowing DEM_FILE over the region of interest
        window = from_bounds(*cell_ext, transform=src.transform)
        str_cell = src.read(1, window=window)  # shape: (bands, height, width)
        arr = np.array(str_cell)
        if not arr.size:
            logger.warning(
                f"Empty cell {id_cell=} detected at {idx=}, probably outside of DEM bounds. Passing."
            )
            continue

        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > 255 or arr.dtype != np.uint8:
            arr -= arr_min
            if arr_max != arr_min:
                arr /= arr_max - arr_min
            arr *= 255
            arr = arr.astype(np.uint8, copy=False)

        # Save image
        img = Image.fromarray(arr, mode="L")  # 'L' = grayscale
        img.save(PIC_PATH, quality=100, optimize=True)
        logger.info(f"Image {id_cell} saved to {PIC_PATH} successfully!")


def process_tile(i, clip_geom):
    """Docstring"""
    CROP_DIR = Path(config.PREDICTION_DIR / f"{i}")
    if not CROP_DIR.exists():
        return
    FOREST_RASTER_MASK_PATH = Path(CROP_DIR / f"forest_raster_mask_{i}.tif")
    # Call tensorflow here using the model file,
    # this takes quite a lot of time, hence the trigger
    if config.RUN_INFERENCE:
        run_inference(config.TF_MODEL_PATH, CROP_DIR)

    logger.info(f"Segmentation executed on forest part {i}")

    postprocess_raster(
        i, CROP_DIR, FOREST_RASTER_MASK_PATH, config.THRESHOLD_SEGMENTATION
    )
    logger.info(f"{CROP_DIR=}")

    if config.USE_SKELETONIZE:
        line_finder_skel(config.PREDICTION_DIR, i, 0.16)
    else:
        line_finder(
            config.PREDICTION_DIR, i, config.THRESH_THINNING, config.WIN_SIZE, clip_geom
        )


# Process a row of a geodataframe: input function for parallel processing
def process_row(i, forest_geom, src):
    """Process a single row geometry of a GeoDataFrame"""
    result = extract_tiles(i, forest_geom)  # process forest geometry in row
    if result:
        geom_crop, wa_transform = result
        build_tile(i, geom_crop, wa_transform, src)
        process_tile(i, forest_geom)


class Worker:
    def __init__(self, dem_filtered_file):
        self.dem_filtered_file = dem_filtered_file
        self.src = None  # will be opened lazily in each worker process

    def __getstate__(self):
        """Called when pickling the Worker to send it to other processes."""
        return {"dem_filtered_file": self.dem_filtered_file}

    def __setstate__(self, state):
        """Called inside each worker process when the Worker is unpickled."""
        self.dem_filtered_file = state["dem_filtered_file"]

        # Open raster once *in the worker process*
        logger.info(f"Opening {config.DEM_FILE_FILTERED=}...")
        self.src = rio.open(self.dem_filtered_file)

    def __call__(self, args):
        i, geom = args
        # Pass src to external functions
        return process_row(i, geom, self.src)

    def __del__(self):
        """Ensure the raster file is closed when the Worker is deleted."""
        if self.src:
            logger.info(f"Closing raster file {config.DEM_FILE_FILTERED}.")
            self.src.close()
            self.src = None


# This matches or equals the "preprocess_raster" function in the raster_processing.R file:
# gdf comes from the forest delineation shapefile (polygon vector data)
def process_gdf(gdf):
    """Process the GeoDataFrame in parallel."""
    worker = Worker(config.DEM_FILE_FILTERED)
    tasks = [(i, row.geometry) for i, row in gdf.iterrows()]

    with ProcessPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor for each row
        for _ in executor.map(worker, tasks):
            pass


# %%
def run():
    """Main function"""
    start = time()
    gdf = gpd.read_file(config.DELINEATION_FILE, layer=config.DELINEATION_LAYER)

    # It's good not to load every feature if you work
    # with unsynchronized coverage between DEM and vector features
    if config.CROP_GDF:
        gdf = crop_gdf_to_available_dem_extent(
            gdf,
            dem_file=config.DEM_FILE_FILTERED,
            intersection=True,
        )

    build_grid(gdf)
    process_gdf(gdf)

    delta_t = time() - start
    logger.info(f"Skid detector run successfully in {delta_t:.2f} seconds.")


# %%
if __name__ == "__main__":
    run()
