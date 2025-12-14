#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: config.py
Author: nicolas-heigvd
Date: 2025-11-03
Version: 1.0
Description: This is the main config script.
"""

# %%
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Docstring for Config
    """

    ROOT_DIR = Path(__file__).resolve().parent.parent

    # Constants and triggers
    BASE_DIR = (ROOT_DIR / Path(os.getenv("APP_DIR", "."))).resolve()
    DATA_DIR = (ROOT_DIR / Path(os.getenv("DATA_DIR", "/data"))).resolve()
    DEMO = os.getenv("DEMO", "false").lower() == "true"
    # Demo mode:
    if DEMO:
        DATA_DIR = DATA_DIR / "DEMO"

    # Fetch swissalti constants and triggers
    LOGLEVEL = os.getenv("LOGLEVEL", "DEBUG")
    DOWNLOAD_TILE = os.getenv("DOWNLOAD_TILE", "false").lower() == "true"
    PROCESS_TILE_IMG = os.getenv("PROCESS_TILE_IMG", "false").lower() == "true"
    LOG_TAB = "\n\t\t\t\t"

    # Check if the environment variable is set and directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory {DATA_DIR} not found.")

    # Main data directories
    RAW_DATA_DIR = Path(DATA_DIR / "RAW_DATA")
    TEMP_DIR = Path(DATA_DIR / "TEMP")
    PROCESSED_DIR = Path(DATA_DIR / "PROCESSED")

    # Child data directories
    INPUT_SKID_DIR = Path(RAW_DATA_DIR / "SKID")
    TEMP_DEM_DIR = Path(TEMP_DIR / "DEM")
    DELINEATION_DIR = Path(TEMP_DIR / "DELINEATION")
    PREDICTION_DIR = Path(TEMP_DIR / "PREDICTION")
    INPUT_DEM_DIR = Path(TEMP_DEM_DIR / "TILES")

    # Processed directory
    OUTPUT_DEM_DIR = Path(PROCESSED_DIR / "DEM")
    OUTPUT_DATA_DIR = Path(PROCESSED_DIR / "SKID")

    # Specific files pointers
    DELINEATION_LAYER = "gdf_forest_clipped_path"
    DELINEATION_FILE = Path(DELINEATION_DIR / f"{DELINEATION_LAYER}.gpkg")
    # File containing list of tiles as fetched from geo.admin OGD API
    TILES_LIST_FILENAME = "ch.swisstopo.swissalti3d-vaud.csv"
    if DEMO:
        TILES_LIST_FILENAME = "ch.swisstopo.swissalti3d-demo.csv"

    TILES_LIST_FILE = Path(DATA_DIR / "RAW_DATA" / TILES_LIST_FILENAME)
    DEM_FILE = Path(TEMP_DEM_DIR / "swissalti3d_Vaud_0.5_2056_5728.vrt")
    DEM_FILE_FILTERED = Path(TEMP_DEM_DIR / "swissalti3d_Vaud_0.5_2056_5728_diff.vrt")

    # Define output grid path
    PREDICTION_FILE = Path(PREDICTION_DIR / "grid_id.tif")
    # tf model path
    TF_MODEL_PATH = (BASE_DIR / "model" / "road_finder_model.h5").resolve()

    # Create folder if need be
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DEM_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DEM_DIR.mkdir(parents=True, exist_ok=True)
    DELINEATION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    # General processing triggers
    CLIP_FORESTS = False  # Clip vector features with cantonal boundary, set this to False if the operation was already done in order to load the clipped geopackage
    CROP_GDF = True  # Crop the vector features with the DEM extent

    # Constants for processing tiles with tf
    MIN_FOREST_AREA = 1e4  # 1 ha
    WINDOW_SIZE = 150
    HALF_WINDOW = WINDOW_SIZE / 2
    THRESHOLD_SEGMENTATION = 0.5

    OUTPUT_FILE = Path(OUTPUT_DATA_DIR / "forest_roads")

    VECTORIZE_SEGMENTATION = True
    THRESH_MIN_AREA = 20  # minimum area in m2
    THRESH_THINNING = 7  # minimum number of positive neighbors
    WIN_SIZE = 2.5  # window size for line finder
    MIN_LINE_LENGTH = 30

    RUN_INFERENCE = os.getenv("RUN_INFERENCE", "false").lower() == "true"
    USE_SKELETONIZE = os.getenv("USE_SKELETONIZE", "false").lower() == "true"


# %%
config = Config()
