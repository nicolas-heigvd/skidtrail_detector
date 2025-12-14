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

from datetime import datetime
from time import time

from logging_config import logger
from skidtrail_detector.dtmanalyzer import run as run_dtm_analyzer
from skidtrail_detector.fetch_swissalti import run as run_fs
from skidtrail_detector.skid_detector import run as run_skidtrail_detector

# %%
start = time()
start_time_str = datetime.fromtimestamp(start).isoformat(timespec="seconds")
logger.info(f"Main script launched at {start_time_str}")
run_fs()
run_dtm_analyzer()
run_skidtrail_detector()

delta_t = time() - start
logger.info(f"Main script run successfully in {delta_t:.2f} seconds.")
