#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: dtmanalyzer.py
Author: nicolas-heigvd
Date: 2025-08-22
Version: 1.0
Description: This script is a conversion from the R script dtmanalyzer.R
"""

# %%
from pathlib import Path

import defusedxml.ElementTree as ET
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter

from config import config
from logging_config import logger
from skidtrail_detector.raster_processing import build_vrt_file

# Check if the environment variable is set and directory exists
if not config.DATA_DIR.exists():
    raise FileNotFoundError(f"Data directory {config.DATA_DIR} not found.")


# %%
def load_raster(filename):
    """Load raster file and return numpy array"""
    with rio.open(filename) as src:
        data = src.read(1)  # Read first band
        profile = src.profile

    return data, profile


def bounds_to_window(bounds, transform):
    """Helper function to build window from bounds and affine transform"""
    xmin, ymin, xmax, ymax = bounds

    return from_bounds(xmin, ymin, xmax, ymax, transform)


def crop_to_inner_window(win_pad, win_orig, padded_array):
    """
    Crop a padded array to the original tile size, handling border tiles.

    Parameters
    ----------
    win_pad : Window
        The padded window requested (may extend outside raster)
    win_orig : Window
        The original tile window (unpadded)
    padded_array : np.ndarray
        Array actually read (may be smaller at borders)

    Returns
    -------
    cropped : np.ndarray
        Cropped array of size <= original tile
    """
    row_off = int(max(0, win_orig.row_off - win_pad.row_off))
    col_off = int(max(0, win_orig.col_off - win_pad.col_off))

    # Compute how many rows/cols are available in padded_array
    max_rows, max_cols = padded_array.shape
    nrows = min(int(win_orig.height), max_rows - row_off)
    ncols = min(int(win_orig.width), max_cols - col_off)

    return padded_array[row_off : row_off + nrows, col_off : col_off + ncols]


def save_raster(filename, data, profile):
    """Save raster data from a numpy array to a file on disk"""
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=profile["crs"],
        transform=profile["transform"],
        nodata=profile["nodata"],
    ) as dst:
        dst.write(data, 1)


def filter_array(data: np.array) -> np.array:
    """Apply gaussian filter and normalization on a numpy array"""
    # Convert to image format and handle NaNs (set NaNs or other invalid data to 0)
    dem_img = np.ma.masked_invalid(data).filled(0)

    # Smoothing using Gaussian filter (equivalent to isoblur)
    dem_iso = gaussian_filter(dem_img, sigma=3, mode="nearest")

    # Subtract smoothed DTM from original
    dem_iso = dem_img - dem_iso

    # Set values < -1 to -1 and > 1 to 1
    dem_iso = np.clip(dem_iso, -1, 1)

    return dem_iso


def load_cells_from_gpkg(
    gpkg_path,
    cell_layer="cell_polygons",
    padded_layer="padded_cell_polygons",
    center_layer="cell_center_points",
    add_bounds=True,
):
    """
    Load cell polygons, padded polygons, and center points from a GeoPackage,
    merge them into one GeoDataFrame, and use the original polygons as geometry.

    Assumes all layers share the same index.
    """

    # Load layers
    gdf_cells = gpd.read_file(gpkg_path, layer=cell_layer)
    gdf_padded = gpd.read_file(gpkg_path, layer=padded_layer)

    # Optional center points
    gdf_centers = None
    if center_layer is not None:
        gdf_centers = gpd.read_file(gpkg_path, layer=center_layer)

    # Prepare final GDF using original polygons as geometry
    gdf = gdf_cells.copy()  # original cells = active geometry

    # Rename geometry columns to avoid conflicts
    gdf_padded = gdf_padded.rename_geometry("padded_geom")

    if gdf_centers is not None:
        gdf_centers = gdf_centers.rename_geometry("center_geom")

    # Join (shared index allows direct join)
    gdf = gdf.join(gdf_padded["padded_geom"])

    if gdf_centers is not None:
        gdf = gdf.join(gdf_centers["center_geom"])

    # Add numeric bounds for fast raster I/O
    if add_bounds:
        gdf["orig_bounds"] = gdf.geometry.bounds[
            ["minx", "miny", "maxx", "maxy"]
        ].values.tolist()
        gdf["pad_bounds"] = (
            gdf["padded_geom"].bounds[["minx", "miny", "maxx", "maxy"]].values.tolist()
        )

    return gdf


def filename_to_index(filename: str) -> str:
    """
    Extract the tile index from a GeoTIFF filename.
    """
    fname = Path(filename).stem  # remove extension
    # Split by '_' and find the part that looks like '2544-1197'
    for part in fname.split("_"):
        if "-" in part:
            return part.replace("-", "_")
    raise ValueError(f"No tile index found in filename {filename}")


def check_padding(pad_bounds, src_bounds):
    """Check if paddings is outside source boundaries"""
    left_src, bottom_src, right_src, top_src = src_bounds
    left_pad, bottom_pad, right_pad, top_pad = pad_bounds
    left_pad = left_src if left_pad < left_src else left_pad
    bottom_pad = bottom_src if bottom_pad < bottom_src else bottom_pad
    right_pad = right_src if right_pad > right_src else right_pad
    top_pad = top_src if top_pad > top_src else top_pad

    return left_pad, bottom_pad, right_pad, top_pad


def filter_tiles(dem_file: Path = config.DEM_FILE):
    """
    gdf must contain:
        - orig_bounds: original 1km tile bounds
        - pad_bounds: padded bounds
    """
    kartenblatt_cell_path = config.TEMP_DEM_DIR / "kartenblatt_cell_path.gpkg"
    gdf = load_cells_from_gpkg(
        gpkg_path=kartenblatt_cell_path,
        cell_layer="cell_polygons",
        padded_layer="padded_cell_polygons",
        center_layer=None,
    )

    tree = ET.parse(dem_file)
    root = tree.getroot()
    source_files = [Path(element.text) for element in root.findall(".//SourceFilename")]
    tile_file_map = {filename_to_index(f.name): f for f in source_files}
    gdf["source_file"] = gdf["index"].map(lambda idx: tile_file_map.get(idx))
    output_dir = config.INPUT_DEM_DIR / "DIFF"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    with rio.open(dem_file) as src:
        profile = src.profile
        transform = src.transform

        for _, row in gdf.iterrows():
            source_file = row["source_file"]
            if source_file is None:
                logger.info(f"No source file found for tile {row['index']}, skipping")
                continue
            orig_bounds = row["orig_bounds"]
            pad_bounds = row["pad_bounds"]
            pad_bounds = check_padding(pad_bounds, src.bounds)

            # Steps
            # A. Compute windows
            win_orig = bounds_to_window(orig_bounds, transform)
            win_pad = bounds_to_window(pad_bounds, transform)

            # B. Read padded data
            padded = src.read(1, window=win_pad)

            # C. Apply your filter
            filtered = filter_array(padded)

            # D. Crop back to original extent
            cropped = crop_to_inner_window(win_pad, win_orig, filtered)

            # E. Prepare output profile
            out_profile = profile.copy()
            out_profile.update(
                driver="GTiff",
                height=win_orig.height,
                width=win_orig.width,
                transform=rio.windows.transform(win_orig, transform),
                count=1,
                dtype=cropped.dtype,
            )

            # F. Save result tile
            base_name = Path(source_file).stem
            tile_filename = Path(output_dir) / f"{base_name}_diff.tif"
            with rio.open(tile_filename, "w", **out_profile) as dst:
                dst.write(cropped, 1)

            logger.info(f"Saving {tile_filename=}...")

    return output_dir


def run():
    """Main function"""
    if not config.PROCESS_TILE_IMG:
        logger.info(f"DEM tiles won't be processed because {config.PROCESS_TILE_IMG=}.")
        return
    filtered_dem_dir = filter_tiles(dem_file=config.DEM_FILE)
    build_vrt_file(filtered_dem_dir)

    logger.info(f'Script "{__file__}" run successfully!')


# %%
if __name__ == "__main__":
    run()

# %%
