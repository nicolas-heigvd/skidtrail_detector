#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:43:00 2024
# Copyright (C) 2024-present {nicolas.blanc} @ HEIG-VD
# This file is licensed under the GPL-3.0-only. See LICENSE file for details.
# Third-party libraries and their licenses are listed in the NOTICE.md file.
"""

# %%
import re
from pathlib import Path, PosixPath
from time import time

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import requests
from dotenv import load_dotenv
from geopandas import GeoDataFrame
from lxml import etree
from shapely.geometry import Point, Polygon, box

from config import config
from logging_config import logger
from skidtrail_detector.raster_processing import build_vrt_file, get_extent_from_profile

load_dotenv()

# %%
# URLs
geoadmin_url = "https://data.geo.admin.ch"
getcapabilities_url = (
    "https://wms.geo.admin.ch/en/?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities"
)

qurl = (
    "https://wms.geo.admin.ch/en/?SERVICE=WMS"
    "&VERSION=1.3.0"
    "&REQUEST=GetFeatureInfo"
    "&BBOX={bbox}"
    "&CRS=EPSG:{srid}"
    "&WIDTH=1000"
    "&HEIGHT=1000"
    "&LAYERS=ch.swisstopo.images-swissimage-dop10.metadata-kartenblatt"
    "&STYLES=Default"
    "&FORMAT=image%2Fjpeg"
    "&QUERY_LAYERS=ch.swisstopo.images-swissimage-dop10.metadata-kartenblatt"
    "&INFO_FORMAT={info_format}"
    "&I=500"
    "&J=500"
    "&FEATURE_COUNT=1"
)


# %%
def load_layer_from_gpkg(gpkg_filepath: PosixPath, layer_name: str) -> GeoDataFrame:
    """Load an existing layer from a GeoPackage file on disk to a GeoDataFrame

    Parameters:
    ----------
    gpkg_filepath : PosixPath
        A SWISSTLM3D GeoPackage file path on disk.
    layer_name : str
        The name of a layer in the GeoPackage file.

    Returns:
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the resulting data.
    """
    gdf = None
    layer_names = fiona.listlayers(gpkg_filepath)
    if gpkg_filepath.is_file() and layer_name in layer_names:
        gdf = gpd.read_file(gpkg_filepath, layer=layer_name)
        gdf.set_crs(epsg=2056, allow_override=True, inplace=True)

    return gdf


def filter_gdf(gdf: GeoDataFrame, column_name: str, filter_value: str) -> GeoDataFrame:
    """Filter a GeoDataFrame by appolying a filter_value over a column_name.

    Parameters:
    ----------
    gpkg_filepath : PosixPath
        A SWISSTLM3D GeoPackage file path on disk.
    column_name : str
        The name of a column on which to filter data.
    filter_value : str
        The name of the layer.

    Returns:
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the resulting data.
    """
    return gdf[
        gdf[column_name].str.contains(
            filter_value, flags=re.IGNORECASE, regex=True, na=False
        )
    ]


def load_cantonal_boundary_from_swissboundaries3d_gpkg(
    gpkg_filepath: PosixPath,
    filter_value: str = "Vaud",
) -> GeoDataFrame:
    """Load SWISSBOUNDARIES3D GeoPackage into a GeoDataFrame

    Parameters:
    ----------
    gpkg_filepath : PosixPath
        A SWISSBOUNDARIES3D GeoPackage file path on disk.
    filter_value : str
        The name of the layer.

    Returns:
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the resulting data.
    """
    start = time()
    layer_name = "tlm_kantonsgebiet"
    gdf = load_layer_from_gpkg(gpkg_filepath, layer_name)
    gdf = filter_gdf(gdf, "name", filter_value)
    delta_t = time() - start
    logger.info(f"Canton {filter_value} loaded successfully in {delta_t:.2f} seconds.")

    return gdf


def load_forests_from_swisstlm3d_gpkg(
    gpkg_filepath: PosixPath, filter_value: str = "wald"
) -> GeoDataFrame:
    """Load SWISSTLM3D GeoPackage into a GeoDataFrame

    Parameters:
    ----------
    gpkg_filepath : PosixPath
        A SWISSTLM3D GeoPackage file path on disk.
    filter_value : str
        The name of the layer.

    Returns:
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the resulting data.
    """
    start = time()
    layer_name = "tlm_bb_bodenbedeckung"
    gdf = load_layer_from_gpkg(gpkg_filepath, layer_name)
    gdf = filter_gdf(gdf, "objektart", filter_value)
    delta_t = time() - start
    logger.info(
        f"Forest {list(gdf['objektart'].unique())} loaded successfully in {delta_t:.2f} seconds."
    )

    return gdf


def select_features_from_extent(
    gdf_features: gpd.GeoDataFrame, bbox: np.array, intersection: bool = False
) -> gpd.GeoDataFrame:
    """Select feature in a GeoDataFrame from a bbox.

    Parameters:
    ----------
    gdf_features : GeoDataFrame
        The GeoDataFrame containing the features to clip.
    bbox : rasterio.coords.BoundingBox
        The bbox to select the features

    Returns:
    -------
    gdf_out : GeoDataFrame
        The new GeoDataFrame.
    """
    start = time()
    clip_geom = box(*bbox)
    gdf_filtered = gdf_features[gdf_features.intersects(clip_geom)]
    gdf_out = gdf_filtered.copy()
    if intersection:
        gdf_out["geometry"] = gdf_filtered.intersection(clip_geom)
        gdf_out = gdf_out[~gdf_out.is_empty]

    delta_t = time() - start
    logger.debug(
        f"Selecting features in gdf using {intersection=} took {delta_t:.2f} seconds."
    )

    return gdf_out


def crop_gdf_to_available_dem_extent(
    gdf: gpd.GeoDataFrame, dem_file: Path, intersection: bool = False
):
    """Crop a gdf containing forest shape to the available DEM extent for further processing"""
    with rio.open(dem_file) as src:
        bbox = src.bounds

    new_gdf = select_features_from_extent(gdf.copy(), bbox, intersection)
    outfile = config.DELINEATION_DIR / "gdf_forest_reduced_path.gpkg"
    new_gdf.to_file(outfile, driver="GPKG", index=True)

    return new_gdf


def clip_gdf(
    gdf_clip: GeoDataFrame, gdf_features: gpd.GeoDataFrame, intersection: bool = True
) -> GeoDataFrame:
    """Clip a GeoDataFrame with a clipping region from another.

    Parameters:
    ----------
    gdf_clip : GeoDataFrame
        The clipping GeoDataFrame.
    gdf_features : GeoDataFrame
        The GeoDataFrame containing the features to clip.

    Returns:
    -------
    gdf_clipped : GeoDataFrame
        The clipped GeoDataFrame.
    """
    start = time()
    clip_geom = gdf_clip.geometry.iloc[0]
    gdf_filtered = gdf_features[gdf_features.intersects(clip_geom)]
    gdf_clipped = gdf_filtered.copy()
    if intersection:
        gdf_clipped["geometry"] = gdf_filtered.intersection(clip_geom)
        gdf_clipped = gdf_clipped[~gdf_clipped.is_empty]

    delta_t = time() - start
    logger.debug(f"Clipping gdf using {intersection=} took {delta_t:.2f} seconds.")

    return gdf_clipped


def compute_bbox(gdf: GeoDataFrame) -> Polygon:
    """Compute a bbox out of a GeoDataFrame

    Parameters:
    ----------
    geojson_file : str
        The path of a GeoJSON file on the disk. The file must exist.

    Returns:
    -------
    gdf : Polygon instance
        A Polygon object containing the boundary shape of the entire GeoDataFrame.
    """

    return box(*gdf.total_bounds)


def extract_bbox_bounds(bbox: Polygon, format: str = "dict") -> dict:
    """Extract bbox bounds as a dict

    Parameters:
    ----------
    bbox : Polygon
        A boundary shape given as a Polygon.

    Returns:
    -------
    bounds : dict
        A dictionnary containing the min and max coordinates of the given Polygon
        along x and y axis.
    """
    bounds = bbox.bounds
    if format == "dict":
        return {
            "xMin": bounds[0],
            "yMin": bounds[1],
            "xMax": bounds[2],
            "yMax": bounds[3],
        }
    if format == "array":
        return np.array(bounds)
    if format == "str":
        return ",".join([str(x) for x in bounds])


def create_padded_cell_bounds(cell_bounds, pad_m, vrt_bounds):
    x_min_vrt, y_min_vrt, x_max_vrt, y_max_vrt = vrt_bounds

    padded_bounds = np.column_stack(
        [
            np.maximum(cell_bounds[:, 0] - pad_m, x_min_vrt),  # clip to left edge
            np.maximum(cell_bounds[:, 1] - pad_m, y_min_vrt),  # clip to bottom edge
            np.minimum(cell_bounds[:, 2] + pad_m, x_max_vrt),  # clip to right edge
            np.minimum(cell_bounds[:, 3] + pad_m, y_max_vrt),  # clip to top edge
        ]
    )
    return padded_bounds


def create_grid(bbox: Polygon) -> GeoDataFrame:
    """Create a regular grid of point on the center of swiss kilometric tiles"""
    x_min, y_min, x_max, y_max = bbox.bounds
    x_min, y_min = 1e3 * np.floor(np.array([x_min, y_min]) / 1e3)
    x_max, y_max = 1e3 * np.ceil(np.array([x_max, y_max]) / 1e3)
    pixel_size = 0.5
    cell_size = 1000  # meters
    pad_px = 100  # padding amount in pixels
    pad_m = pad_px * pixel_size
    # Generate the x and y coordinates for the grid of cell centers using np.meshgrid
    x_coords = np.arange(x_min + pixel_size * cell_size, x_max, cell_size, dtype=int)
    y_coords = np.arange(y_min + pixel_size * cell_size, y_max, cell_size, dtype=int)

    # Create the meshgrid of coordinates
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    cell_centers = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    # Compute bounds for each cell
    half_pixel_size = pixel_size * cell_size
    # Each row: [x_min, y_min, x_max, y_max]
    cell_bounds = np.column_stack(
        [
            cell_centers[:, 0] - half_pixel_size,  # x_min
            cell_centers[:, 1] - half_pixel_size,  # y_min
            cell_centers[:, 0] + half_pixel_size,  # x_max
            cell_centers[:, 1] + half_pixel_size,  # y_max
        ]
    )
    rounded_bounds = (x_min, y_min, x_max, y_max)
    padded_cells_bounds = create_padded_cell_bounds(cell_bounds, pad_m, rounded_bounds)

    return cell_centers, cell_bounds, padded_cells_bounds


def create_grid_gdf(bbox: Polygon) -> GeoDataFrame:
    """Create a regular grid of point on the center of swiss kilometric tiles

    ----------
    bbox : Polygon
        A bbox square Polygon containing the bounds between which to build
        the array of points in order to query the ogd.swisstopo.admin.ch service.

    Returns:
    -------
    gdf_points : GeoDataFrame instance
        A GeoDataFrame containing the array of points in order to query the
        ogd.swisstopo.admin.ch services.
    """
    cell_centers, cell_bounds, padded_cells_bounds = create_grid(bbox)
    # Create a list of Point objects
    cell_center_points = [Point(x, y) for x, y in cell_centers]
    # Cell polygons from bounds
    cell_polygons = [
        Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
        for x_min, y_min, x_max, y_max in cell_bounds
    ]
    padded_cell_polygons = [
        Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
        for x_min, y_min, x_max, y_max in padded_cells_bounds
    ]
    # Create a GeoDataFrame
    gdf_centers = gpd.GeoDataFrame(geometry=cell_center_points, crs="EPSG:2056")
    gdf_cells = gpd.GeoDataFrame(geometry=cell_polygons, crs="EPSG:2056")
    gdf_cells_padded = gpd.GeoDataFrame(geometry=padded_cell_polygons, crs="EPSG:2056")
    # gdf_cells["year"] =
    gdf_centers.index = [f"{int(x // 1000)}_{int(y // 1000)}" for x, y in cell_centers]
    gdf_cells.index = [f"{int(x // 1000)}_{int(y // 1000)}" for x, y in cell_centers]
    gdf_cells_padded.index = [
        f"{int(x // 1000)}_{int(y // 1000)}" for x, y in cell_centers
    ]
    kartenblatt_cell_path = config.TEMP_DEM_DIR / "kartenblatt_cell_path.gpkg"
    gdf_centers.to_file(kartenblatt_cell_path, driver="GPKG", layer="cell_centers")
    gdf_cells.to_file(kartenblatt_cell_path, driver="GPKG", layer="cell_polygons")
    gdf_cells_padded.to_file(
        kartenblatt_cell_path, driver="GPKG", layer="padded_cell_polygons"
    )

    return gdf_centers, gdf_cells, gdf_cells_padded


def parse_getcap_url(url: str) -> dict:
    """Parse GetCapabilities URL..."""
    response = requests.get(getcapabilities_url, timeout=(5, 30))
    # Check if the request was successful
    if response.status_code == 200:
        try:
            # Parse the XML response using lxml.XML (handles encoding declaration properly)
            root = etree.XML(response.content)

            # Get namespaces from the root element (if any)
            namespaces = root.nsmap.copy()
            default_ns_uri = namespaces.pop(None, None)
            if default_ns_uri:
                namespaces["wms"] = default_ns_uri  # assign a temporary prefix 'wms'

            # XPath using the dynamic prefix
            get_feature_info = root.xpath(
                "//wms:Capability/wms:Request/wms:GetFeatureInfo", namespaces=namespaces
            )

            if get_feature_info:
                element = get_feature_info[0]

                # Initialize dictionary with tag and attributes
                feature_info_dict = {
                    "tag": etree.QName(element).localname,
                    "attributes": dict(element.attrib),
                }

                # Dynamically add child elements
                for child in element:
                    if child.text:
                        tag_name = etree.QName(child).localname
                        if tag_name in feature_info_dict:
                            feature_info_dict[tag_name].append(child.text.strip())
                        else:
                            feature_info_dict[tag_name] = [child.text.strip()]

                logger.debug(f"GetFeatureInfo as dict: {feature_info_dict}")

            else:
                logger.info("GetFeatureInfo element not found.")

        except etree.XMLSyntaxError as e:
            logger.error(f"Error parsing XML: {e}")
    else:
        logger.info(f"Request failed with status code {response.status_code}")

    return feature_info_dict


def query_getfeatureinfo(params: dict, info_format: str) -> True:
    """Query the GetFeatureInfo endpoint

    Parameters:
    ----------
    params : dict
        A dictionary containing query params to build an URL in order to query
        the ogd.swisstopo.admin.ch service.
    """
    bbox_str = ",".join([str(x) for x in params["bounds"]])
    srid = params["srid"]
    # Format the template URL
    url = qurl.format(bbox=bbox_str, srid=srid, info_format=info_format)
    response = requests.get(url, timeout=(5, 30))
    logger.debug(80 * "#")
    logger.debug(f"Querying URL with format: {info_format}")
    logger.debug(f"{url=}")
    logger.debug(f"{response.json()=}")

    return response


def debug(params):
    """Debug function"""
    response = parse_getcap_url(getcapabilities_url)
    formats = response["Format"]
    for info_format in formats:
        query_getfeatureinfo(params, info_format)


def check_format_in_get_featureinfo(info_format: str) -> bool:
    """Check info format from getcap"""
    feature_info_dict = parse_getcap_url(getcapabilities_url)
    formats = feature_info_dict["Format"]
    if info_format in formats:
        logger.info(f'"{info_format}" is available')
        return True
    else:
        return False


def extract_properties(info_format: dict, params: dict):
    """Extract properties from the WMS GetFeatureInfo response"""
    logger.info(f"{params=}")
    response = query_getfeatureinfo(params, info_format).json()
    logger.info(f"{response=}")
    try:
        properties = response["features"][0]["properties"]
        logger.info(f"{properties=}")
    except Exception as err:
        logger.error(f"Unexpected {err=}, {type(err)=}")
        pass

    return properties


def build_tile_name(info_format: str, params: dict):
    """Build tile name from params dictionnary"""
    # tile_year is not the same as swissimage, unfortunately...
    # we can try to guess it, but it's tedious...
    tile_id, tile_year, _ = extract_properties(info_format, params).values()
    resolution = params["resolution"]
    tile_id = tile_id.replace("_", "-")
    srid = params["srid"]
    altitude_epsg_code = 5728
    tile_parent = f"swissalti3d_{tile_year}_{tile_id}"
    tile_name = f"{tile_parent}_{resolution}_{srid}_{altitude_epsg_code}.tif"

    return tile_name, tile_parent


def build_tile_url(
    info_format: str, params: dict, dataset: str = "ch.swisstopo.swissalti3d"
) -> str:
    """Build tile URL

    Results must look like the following form (double check hyphens and underscores):
    https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2021_2530-1157/swissalti3d_2021_2530-1157_0.5_2056_5728.tif
    """
    tile_name, tile_parent = build_tile_name(info_format, params)
    dataset = "ch.swisstopo.swissalti3d" if not dataset else dataset
    url = f"{geoadmin_url}/{dataset}"
    tile_url = f"{url}/{tile_parent}/{tile_name}"
    logger.info(f"{tile_url=}")

    return tile_url


def lv95_to_wgs84(point: np.array, scrs: int, tcrs: int) -> np.array:
    """Docstring"""
    from pyproj import Transformer

    scrs = f"EPSG:{scrs}"
    tcrs = f"EPSG:{tcrs}"
    transformer = Transformer.from_crs(scrs, tcrs, always_xy=True)
    projected_point = transformer.transform(*point)

    return projected_point


def search_stac_api_from_bbox(bbox):
    """Docstring"""
    stac_url = "https://data.geo.admin.ch/api/stac/v1/search"
    ll_0 = bbox.bounds[:2]
    ur_0 = bbox.bounds[2:]
    ll_1 = lv95_to_wgs84(ll_0, 2056, 4326)
    ur_1 = lv95_to_wgs84(ur_0, 2056, 4326)
    bbox_wgs84 = ",".join(map(str, ll_1 + ur_1))
    params = {
        "collections": "ch.swisstopo.swissalti3d",
        "bbox": bbox_wgs84,
        "state": "current",
    }

    return requests.get(stac_url, params=params, timeout=(5, 30))


def get_tif_url(assets):
    """Docstring"""
    # Ensure assets is a dictionary, otherwise return None
    if not isinstance(assets, dict):
        return None

    for filename, data in assets.items():
        if filename.endswith(".tif"):
            return data.get("href", [])

    return None


def get_tif_url_from_stac_response(response):
    """Docstring"""
    all_data = []
    while response.status_code == 200:
        data = response.json()
        features = data.get("features", [])
        df_features = pd.DataFrame(features)
        df_features["tif_url"] = df_features["assets"].apply(get_tif_url)
        df_features = df_features[["id", "collection", "tif_url"]]
        all_data.append(df_features)

        # Check if there is a "next" page in the API response
        links = data.get("links", [])
        next_page_url = None
        for link in links:
            if link.get("rel") == "next":
                next_page_url = link.get("href")
                break

        if next_page_url:
            logger.info(f"Fetching next URL {next_page_url}...")
            response = requests.get(next_page_url, timeout=(5, 30))
        else:
            break

    if all_data:
        df_final = pd.concat(all_data, ignore_index=True)
        df_final[["dir", "filename"]] = df_final["tif_url"].str.extract(
            r"/([^/]+)/([^/]+)$"
        )
        df_final["year"] = df_final["filename"].str.extract(r"_(\d{4})_")[0].astype(int)
        df_final["coords"] = df_final["filename"].str.extract(r"_\d{4}_(\d{4}-\d{4})_")[
            0
        ]
        idx = df_final.groupby("coords")["year"].idxmax()
        df_latest = df_final.loc[idx].reset_index(drop=True)

        return df_latest
    else:
        return pd.DataFrame()


def download_swisstile(fetch_url: str, TEMP_DIR: PosixPath) -> PosixPath:
    """Download a tile asset to a file on disk

    Parameters:
    ----------
    fetch_url : str
        An URL to query a tile from the ogd.swisstopo.admin.ch services.
    TEMP_DIR: PosixPath instance
        A PosixPath object to a temporary folder on disk where to store the
        resulting tile.

    Returns:
    -------
    filename : PosixPath instance
        A PosixPath object to a tile file on disk.
    """
    filename = fetch_url.split("/")[-1]
    filename = Path(config.INPUT_DEM_DIR, filename)
    logger.info(f"Downloading URL {fetch_url} into {filename}...")
    response = requests.get(fetch_url, timeout=(5, 30))
    logger.info(f"{fetch_url} status code [{response.status_code}]")
    if response.status_code != 200:
        return

    with open(filename, mode="wb") as f:
        f.write(response.content)

    return filename


def download_tiles(info_format: str, params: dict, bounds_array: np.array):
    """Docstring"""
    logger.warning(
        "Downloading tiles from online resources, this may take a while "
        "and it can be limited or blocked in any way by the API. "
        "Please make sure you accept the condition of the online ressource "
        f"when using {config.DOWNLOAD_TILE=}."
    )
    for bounds in bounds_array:
        # check point winding order [[x_min, y_min],[x_max, y_max]]
        params["bounds"] = bounds
        fetch_url = build_tile_url(info_format, params)
        download_swisstile(fetch_url, config.INPUT_DEM_DIR)


def download_tiles_from_file(tiles_list_file: Path = config.TILES_LIST_FILE):
    """Download swissalti3d tiles from a csv file"""
    if not tiles_list_file.exists():
        return

    with open(tiles_list_file, "r", encoding="utf8", newline="\n") as f:
        data = f.readlines()

    for fetch_url in data:
        fetch_url = fetch_url.replace("\n", "")
        download_swisstile(fetch_url, config.INPUT_DEM_DIR)


def load_data_from_files():
    """Load cantonal boundary and forest vector data"""
    logger.info("Loading forest data and cantonal boundary...")
    swissboundaries3d_gpkg_filepath = (
        config.RAW_DATA_DIR
        / "swissboundaries3d_2025-04_2056_5728.gpkg/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg"
    )
    swisstlm3d_gpkg_filepath = (
        config.RAW_DATA_DIR
        / "swisstlm3d_2025-03_2056_5728.gpkg"
        / "SWISSTLM3D_2025.gpkg"
    )
    gdf_canton = load_cantonal_boundary_from_swissboundaries3d_gpkg(
        swissboundaries3d_gpkg_filepath, filter_value="Vaud"
    )
    gdf_forest = load_forests_from_swisstlm3d_gpkg(
        swisstlm3d_gpkg_filepath, filter_value="wald"
    )

    return gdf_canton, gdf_forest


def process_forest_data():
    """Process forest data using cantonal boundary"""
    gdf_canton, gdf_forest = load_data_from_files()
    start = time()
    gdf_forest_clipped = clip_gdf(gdf_canton, gdf_forest)
    delta_t = time() - start
    logger.info(f"Forest clipped successfully in {delta_t:.2f} seconds.")
    gdf_forest_clipped_path = config.DELINEATION_FILE
    gdf_forest_clipped.to_file(gdf_forest_clipped_path, driver="GPKG", index=True)
    compute_bbox_from_shape = True
    if compute_bbox_from_shape:
        bbox = compute_bbox(
            gdf_forest_clipped
        )  # roughly equals to the cantonal boundary
    else:  # Getting bbox from DEM data
        with rio.open(config.DEM_FILE) as src:
            profile = src.profile
        bbox = box(*get_extent_from_profile(profile))

    gdf_centers, gdf_cells, gdf_cells_padded = create_grid_gdf(bbox)
    # apply buffer using grid size to take a cell into account even if its centroid lies outside the clipping polygon
    start = time()
    gdf_centers_clipped = clip_gdf(
        gdf_canton.buffer(np.sqrt(2 * 1000**2) / 2), gdf_centers, intersection=True
    )
    delta_t = time() - start
    logger.info(f"Cell centers clipped successfully in {delta_t:.2f} seconds.")
    gdf_centers_clipped_path = config.TEMP_DEM_DIR / "gdf_centers_clipped.gpkg"
    gdf_centers_clipped.to_file(gdf_centers_clipped_path, driver="GPKG")

    start = time()
    gdf_cells_clipped = clip_gdf(gdf_canton, gdf_cells, intersection=False)
    delta_t = time() - start
    logger.info(f"Cells clipped successfully in {delta_t:.2f} seconds.")
    gdf_cells_clipped_path = config.TEMP_DEM_DIR / "gdf_cells_clipped.gpkg"
    gdf_cells_clipped.to_file(gdf_cells_clipped_path, driver="GPKG")

    bounds = extract_bbox_bounds(bbox, format="array")
    logger.info(f"{bounds=}")

    # TODO: build URL with a grid covering the region build on the known bounds
    # because if the region is too large, the API returns an empty result !
    # params.update(bounds)
    # TODO: update logic to parse all points in gdf_grid
    # centers_clipped = get_coordinates(gdf_centers_clipped.geometry)
    # bounds_clipped = np.hstack([centers_clipped - 500, centers_clipped + 500])
    cells_clipped = gdf_cells_clipped.bounds.to_numpy()

    return cells_clipped


def load_clipped_cells_from_file(gdf_cells_clipped_path: Path):
    """Load clipped cells from file on disk"""
    if not gdf_cells_clipped_path:
        gdf_cells_clipped_path = config.TEMP_DEM_DIR / "gdf_cells_clipped.gpkg"

    gdf_cells_clipped = gpd.read_file(gdf_cells_clipped_path)

    return gdf_cells_clipped.bounds.to_numpy()


def prepare_query_and_download_tiles():
    """Prepare query and request geo admin API"""
    info_format = "application/json"
    params = {
        "format": "image/tiff; application=geotiff; profile=cloud-optimized",  # spaces are important here!
        "resolution": 0.5,
        "srid": 2056,
        "state": "current",
    }

    if not check_format_in_get_featureinfo(info_format):
        return

    if config.CLIP_FORESTS:
        logger.info(
            "Clipping forest dataset with cantonal boundary, this make take a while..."
        )
        cells_clipped = process_forest_data()
    else:
        gdf_cells_clipped_path = config.TEMP_DEM_DIR / "gdf_cells_clipped.gpkg"
        logger.info(
            f"Loading clipped forest from existing geopackage: {gdf_cells_clipped_path}"
        )
        cells_clipped = load_clipped_cells_from_file(gdf_cells_clipped_path)

    if config.DOWNLOAD_TILE:
        if config.TILES_LIST_FILE.exists():
            logger.info(f"Downloading tiles using {config.TILES_LIST_FILE=}...")
            download_tiles_from_file(config.TILES_LIST_FILE)
        else:
            # Buggy, year is not the same as the swissimage dataset unfortunately...
            logger.info("Downloading tiles individually from geo admin API...")
            download_tiles(info_format, params, cells_clipped)
    else:
        logger.info(
            f"Tiles will NOT be downloaded from online resource because {config.DOWNLOAD_TILE=}."
        )

    build_vrt_file(config.INPUT_DEM_DIR)

    # dem_filenames = [
    #     elem
    #     for elem in Path(config.TEMP_DEM_DIR).glob("swissalti3d*.tif")
    #     if "clipped" not in elem.name
    # ]


def run():
    """Process forest data and download tiles from swisstopo's API"""
    prepare_query_and_download_tiles()
    logger.info(f'Script "{__file__}" run successfully!')


# %%
if __name__ == "__main__":
    run()
# %%
