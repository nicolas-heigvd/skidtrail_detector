#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: line_finder.py
Author: nicolas-heigvd
Date: 2025-08-22
Version: 1.0
Description: This script is a conversion from the R script line_finder.R
"""

# %%

# import shutil
from pathlib import Path
from time import time

# import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeoops
import rasterio
from rasterio.features import shapes
from rasterio.transform import xy
from scipy.ndimage import generic_filter, uniform_filter
from scipy.spatial import KDTree
from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from config import config

# from error_handling import check_file
from logging_config import logger

# from skimage.morphology import medial_axis, thin


logger.debug(f"Pandas version: {pd.__version__}")


# %%
# Functions
def point_finder(x: np.array) -> int:
    """Find point in 5x5 array"""
    if np.isnan(x[12]) or x[12] == 0:
        return 0
    m = np.array(x).reshape(5, 5)
    upper_sum = np.nansum(m[:2, :])
    lower_sum = np.nansum(m[3:, :])
    right_sum = np.nansum(m[:, 3:])
    left_sum = np.nansum(m[:, :2])
    return int(
        (upper_sum == 0 and right_sum == 0)
        or (upper_sum == 0 and left_sum == 0)
        or (lower_sum == 0 and right_sum == 0)
        or (lower_sum == 0 and left_sum == 0)
    )


def extract_lines(coords_start_0, coords_todo_0, win_size):
    """Docstring"""
    lines = []
    coords_start = coords_start_0.copy()
    coords_todo = coords_todo_0.copy()
    logger.info(f"{coords_start_0.shape=}")
    logger.info(f"{coords_todo_0.shape=}")
    # Build initial KDTree
    tree = KDTree(coords_todo)
    used = np.zeros(len(coords_todo), dtype=bool)
    while len(coords_start) > 2:
        if False:
            coord_center = coords_start[0]
            line = pd.DataFrame([coord_center], columns=["x", "y"])
            coords_start = coords_start[1:]
            while True:
                dis = np.sqrt(np.sum((coords_todo - coord_center) ** 2, axis=1))
                bool_sel = dis <= win_size
                if bool_sel.sum() > 1:
                    # Select points where bool_sel is True
                    coords_sel = coords_todo[bool_sel]
                    # Calculate mean x and y
                    coord_center = coords_sel.mean(axis=0)
                    line.loc[len(line)] = coord_center
                    # Remove selected points from coords_todo
                    coords_todo = coords_todo[~bool_sel]
                else:
                    if len(line) > 1:
                        lines.append(line)
                    break
        elif True:
            coord_center = coords_start[0]
            # coords_start = coords_start[1:]
            line = [coord_center]
            while True:
                # Find all neighbors within win_size
                idxs = tree.query_ball_point(coord_center, r=win_size)
                idxs = [i for i in idxs if not used[i]]  # remove already used points
                if len(idxs) > 1:
                    # Mark as used
                    used[idxs] = True
                    # Get selected coordinates
                    coords_sel = coords_todo[idxs]
                    # Compute new center
                    coord_center = coords_sel.mean(axis=0)
                    line.append(coord_center)
                else:
                    if len(line) > 1:
                        lines.append(pd.DataFrame(line, columns=["x", "y"]))
                    break

        coords_start = coords_start[1:]

    return lines, coords_todo


def extract_lines4(coords_start_0, coords_todo_0, win_size):
    """Docstring"""
    lines = []
    coords_start = coords_start_0.copy()
    coords_todo = coords_todo_0.copy()
    tree = KDTree(coords_todo)
    used = np.zeros(len(coords_todo), dtype=bool)
    for coord_center in coords_start:
        idx_center = tree.query(coord_center, k=1)[1]
        if used[idx_center]:
            continue
        line = [coord_center]
        while True:
            # Find all neighbors within win_size
            idxs = tree.query_ball_point(coord_center, r=win_size)
            idxs = [i for i in idxs if not used[i]]
            if len(idxs) > 1:
                used[idxs] = True
                coords_sel = coords_todo[idxs]
                coord_center = coords_sel.mean(axis=0)
                line.append(coord_center)
            else:
                if len(line) > 1:
                    lines.append(pd.DataFrame(line, columns=["x", "y"]))
                break

    return lines, coords_todo


def write_lines_to_file(all_lines, clip_geom, path_folder, i):
    """Docstring"""
    # logger.info(f"{all_lines=}")
    if len(all_lines) > 0:
        geometries = []
        ids = []

        for idx, df in enumerate(all_lines):
            # Extract (x, y) tuples and create LineString
            coords = df[["x", "y"]].to_numpy()
            line = LineString(coords)
            geometries.append(line)
            # logger.info(f"{idx=}")
            ids.append(idx + 1)  # Match R's 1-based indexing

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {"id": ids, "geometry": geometries}, crs="EPSG:2056"
        )  # Adjust CRS if needed
        clipped_lines_gdf = gdf.clip(clip_geom)
        clipped_lines_gdf = clipped_lines_gdf[
            (clipped_lines_gdf.geometry.type == "LineString")
            & (clipped_lines_gdf.geometry.length > config.MIN_LINE_LENGTH)
        ]
        # Define output file path
        output_file_path = path_folder / f"lines_{i}.shp"
        # Save to Shapefile
        clipped_lines_gdf.to_file(
            output_file_path, driver="ESRI Shapefile", index=False
        )

        logger.info(f"Lines writen to {output_file_path} successfully!")

    return


def line_finder(PREDICTION_DIR, i, thresh_thinning, win_size, clip_geom):
    """Docstring"""
    crop_dir = Path(PREDICTION_DIR) / str(i)
    segmentation_results_path = Path(crop_dir) / f"segmentation_results_{i}.tif"
    # logger.debug(f"{crop_dir=}")
    # logger.debug(f"{segmentation_results_path=}")

    if not segmentation_results_path.exists():
        logger.warning(f"ERROR: {segmentation_results_path} does not exist.")
        return False

    start_time = time()
    with rasterio.open(segmentation_results_path, "r") as src:
        seg = src.read()
        transform = src.transform
        # profile = src.profile
    # Thinning of raster
    focal_sum = uniform_filter(
        seg.astype(float), size=win_size, mode="constant", cval=0.0
    )
    focal_sum *= win_size**2

    # Thresholding
    seg_f1 = np.where(focal_sum < thresh_thinning, 0, 1).astype(np.uint8)
    # seg_f1 must be a 2D numpy array
    seg_f1 = seg.copy()[0]

    # with rasterio.open('/media/nicolas/Data/layons/dtmanalyzer/data/TEMP/PREDICTION/0/seg_f1.tif', "w", **profile) as dst:
    #    dst.write(seg_f1, 1)

    footprint = np.ones((5, 5))  # 5x5 window of ones
    # TODO: check this out
    start_points = generic_filter(
        seg_f1.astype(float),  # input array as float for NaN handling
        function=point_finder,  # your custom function defined earlier
        footprint=footprint,  # moving window shape
        mode="constant",  # pad edges with a constant value
        cval=np.nan,  # use NaN for padding to mimic R's NA
    )
    # Find indices (row, col) where start_points == 1
    rows, cols = np.where(start_points == 1)
    rows_todo, cols_todo = np.where(seg_f1 == 1)
    # Convert row, col to x, y coordinates
    coords_start = np.array([xy(transform, r, c) for r, c in zip(rows, cols)])
    # All points (pixel centers) of white stripes
    coords_todo = np.array([xy(transform, r, c) for r, c in zip(rows_todo, cols_todo)])
    logger.debug(f"{coords_start.shape=}")
    logger.debug(f"{coords_todo.shape=}")
    # First pass
    lines1, coords_unassigned = extract_lines(coords_start, coords_todo, win_size)
    logger.debug(f"{len(coords_unassigned)=}")

    # Second pass: use remaining coords as both source and target
    lines2, _ = extract_lines(coords_unassigned, coords_unassigned, win_size)
    all_lines = lines1 + lines2
    write_lines_to_file(all_lines, clip_geom, crop_dir, i)
    delta_t = time() - start_time
    logger.info(f"Segmentation finished successfully in {delta_t:.2f} seconds!")


def line_finder_skel(PREDICTION_DIR, i, threshold=0.2):
    """
    Docstring
    Doc: https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_skeleton.html
    """
    crop_dir = Path(PREDICTION_DIR) / str(i)
    segmentation_results_path = Path(crop_dir) / f"segmentation_results_{i}.tif"

    if not segmentation_results_path.exists():
        logger.error(f"ERROR: {segmentation_results_path} does not exist.")
        return False

    start_time = time()
    with rasterio.open(segmentation_results_path, "r") as src:
        # profile = src.profile
        img = src.read(1)
        transform = src.transform
        crs = src.crs

    binary = (img > threshold).astype(np.uint8) * 255
    # thinned = thin(binary).astype(np.uint8) * 255
    skeleton = skeletonize(binary, method="lee").astype(np.uint8) * 255
    # skeleton_bis, _ = medial_axis(binary, return_distance=True)
    labeled = label(skeleton, connectivity=2)
    regions = regionprops(labeled)

    lines = []
    if False:
        # Output: Polygon
        logger.debug("Using shapes to vectorize lines...")
        for coords, value in shapes(skeleton, transform=transform):
            if value > 0:
                lines.append(shape(coords))
        merged_lines = unary_union(lines)
        gdf = gpd.GeoDataFrame(geometry=[merged_lines], crs=crs)
    elif True:
        # Output: Linestring
        logger.debug("Using regions to vectorize lines...")
        for region in regions:
            coords = region.coords  # row, col
            if len(coords) < 2:
                continue
            # Convert (row, col) → (x, y) using raster transform
            coords_xy = [
                transform * (col, row) for row, col in coords
            ]  # convert to (x, y)
            if len(coords_xy) > 1:
                line = LineString(coords_xy)
                if line.length >= 1:  # filter very small ones
                    lines.append(line)

        gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    else:
        # Output: Polygon
        logger.debug("Using pygeoops to vectorize lines...")
        for coords, value in shapes(skeleton, skeleton > 0, transform=transform):
            lines.append(shape(coords))

        gdf = gpd.GeoDataFrame(geometry=lines, crs=src.crs)
        gdf["geometry"] = gdf.buffer(0.2, join_style="mitre")
        gdf = gdf.dissolve().explode()
        # Calculate the centerlines
        gdf["geometry"] = pygeoops.centerline(gdf["geometry"])

    skel_file_path = Path(crop_dir) / "extracted_lines2.shp"
    gdf.to_file(skel_file_path)
    delta_t = time() - start_time
    logger.info(
        f"Segmentation finished successfully in {delta_t:.2f}!\nFile created successfully: {skel_file_path}."
    )


# %%
