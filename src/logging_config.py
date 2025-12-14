#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7 10:24:00 2024
# Copyright (C) 2024-present {nicolas.blanc} @ HEIG-VD
# This file is licensed under the GPL-3.0-only. See LICENSE file for details.
# Third-party libraries and their licenses are listed in the NOTICE.md file.
"""

# %%
import logging
import os

# %%

LOGLEVEL = os.getenv("LOGLEVEL", "INFO")


def set_logger(loglevel: str) -> logging.Logger:
    """This is a convenience function to add logging abilities."""
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(
        fmt="{asctime}: {levelname} - {message}",
        # datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(console_handler)

    return logger


logger = set_logger(LOGLEVEL)

# %%
