#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: api.py
Author: nicolas-heigvd
Date: 2025-11-28
Version: 1.0
Description: A small API for gdal CLI
"""
# %%

import logging
import subprocess  # nosec
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)


# Suppress logging for /health
class HealthFilter(logging.Filter):
    def filter(self, record):
        return "/health" not in record.getMessage()


# Apply filter to werkzeug logger
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.addFilter(HealthFilter())


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/build_vrt", methods=["POST"])
def build_vrt():
    """Build a VRT file from a request
    Call: POST /build_vrt input_pattern "*.tif" output "/output/path"
    e.g. with Python:
    url = 'http://gdal:5001/build_vrt'
    payload = { "input_pattern": "/data/TEMP/DEM/TILES/*.tif", "output": "/data/TEMP/DEM/output.vrt" }
    response = requests.post(url, json=payload)
    """
    data = request.get_json()
    input_pattern = data.get("input_pattern")
    output = data.get("output")

    if not input_pattern or not output:
        return (
            jsonify({"success": False, "error": "Missing input_pattern or output"}),
            400,
        )

    # Convert to Path objects
    output_path = Path(output)
    input_path = Path(input_pattern)
    if input_path.is_absolute():
        # If absolute path with a wildcard in the name
        files = [str(f) for f in input_path.parent.glob(input_path.name) if f.is_file()]
    else:
        # Relative path inside /data
        base_dir = Path("/data")
        files = [str(f) for f in base_dir.glob(input_pattern) if f.is_file()]

    if not files:
        return (
            jsonify({"success": False, "error": f"No files match {input_pattern}"}),
            400,
        )

    cmd = ["gdalbuildvrt", str(output_path)] + files

    process = subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec

    return jsonify(
        {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "cmd": " ".join(cmd),
        }
    )


# %%
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)


# %%
