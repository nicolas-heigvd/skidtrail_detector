# Skid Trail Detector

## Status
![Python Version](https://img.shields.io/badge/Python-3.11.11-blue.svg?logo=python&logoColor=f5f5f5)
[![License: MIT/X Consortium License](https://img.shields.io/github/license/nicolas-heigvd/...)](./LICENSE)
[![Deploy Docker Image](https://github.com/nicolas-heigvd/.../actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/nicolas-heigvd/.../actions/workflows/build.yml)
![Python CI](https://github.com/nicolas-heigvd/.../actions/workflows/deploy.yml/badge.svg?branch=main)



## Introduction

This repository contains the code of a Python application to detect skid trail in forests area.



## Prerequisites

A machine having good hardware componants:
- several dozen gigabytes of RAM
- many CPU cores

A machine having at least the following software requirements is needed to run the code:
- Docker engine 29.1.2
- Docker Compose v5.0.0

You can get Docker [here](https://docs.docker.com/get-started/get-docker/).

### Note to Windows users

Please make sure you are using Docker with [WSL2 (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install) to run Linux containers.


## Preparing a file structure on the host for storing input and output data

In order to run the code on your own data, you need a place outside the home project's folder for storing both input and output data.

To do so, a data folder with the following structure is needed on the host:
```sh
data
└── RAW_DATA
    ├── swissboundaries3d_2025-04_2056_5728.gpkg
    │   └── swissBOUNDARIES3D_1_5_LV95_LN02.gpkg
    └── swisstlm3d_2025-03_2056_5728.gpkg
        └── SWISSTLM3D_2025.gpkg
```

Files "swissBOUNDARIES3D_1_5_LV95_LN02.gpkg" and "SWISSTLM3D_2025.gpkg" are mandatory.


## Setting up the environment

For the code to run smoothly, you need to define a few environment variables in a `.env` file.

To this end, copy the `env.sample` file to `.env`: `cp env.sample .env` in the home project's folder and change the variables defined in this `.env` file according to your own environment.

The `.env` file MUST contain the definition of those 8 variables:
| Variable    | Content description |
| ----------- | ------- |
| `ENV` | MUST be set to either `DEV` or `PROD`. However, the code is not ready yet to be deployed in a production server. |
| `LOGLEVEL` | MUST be one of the officially supported [logging levels](https://docs.python.org/3/library/logging.html#logging-levels), for example `DEBUG` or `INFO`. |
| `DATA_DIR` | MUST be a path to the root of the data folder. It MUST be given as a relative path to the project home directory. |
| `DOWNLOAD_TILE` |  MUST be set to a valid [Python boolean](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) which is either `True` or `False`. Set to `True` only if you want to download the tiles from Internet. It is useful to set this variable to `False` when the tiles have already been downloaded. |
| `PROCESS_TILE_IMG` |  MUST be set to a valid [Python boolean](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) which is either `True` or `False`. Set to `True` only if you want to filter tiles from the original ones. This filtering operation will write new tiles on disk. It is useful to set this variable to `False` when the tiles have already been filtered. |
| `RUN_INFERENCE` | MUST be set to a valid [Python boolean](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) which is either `True` or `False`. If set to `True`, the inference routine wil be run. It is useful to set this variable to `False` if you already have inference results that you want to further process because inference time may be quite long.  |
| `USE_SKELETONIZE` | MUST be set to a valid [Python boolean](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) which is either `True` or `False`. If set to `True` the algorithm will run a skeletonization process to extract vector lines from the raster image. |
| `DEMO` | MUST be set to a valid [Python boolean](https://docs.python.org/3/library/stdtypes.html#boolean-type-bool) which is either `True` or `False`. If set to `True` the algorithm will be run against a the demo data provided with the project. |


Once the `.env` file is correctly set up, you can move on to the next step.


## Test data
There is `./tests/data` folder in the project. It contains some test data that are used to demonstrate the project's feasibility.



## Running the container

Finally, run the docker container using:
```sh
docker compose up -d
```

It can take a little while to fetch the data and build the container the very first time you execute this command, don't worry and grab a coffee ☕! The resulting image is roughly 2 GB.

After running the script, the results will be stored in the `TEMP` folder as follow:

```sh
data
├── RAW_DATA
│   ├── swissboundaries3d_2025-04_2056_5728.gpkg
│   └── swisstlm3d_2025-03_2056_5728.gpkg
└── TEMP
    ├── DELINEATION
    ├── DEM
    │   └── TILES
    │       └── DIFF
    └── PREDICTION
        └── <forest_id>
            ├── masks
            └── pics
```

Where `<forest_id>` is the id of a single forest, so beware as you may end up with several thousand folders.


### Extras

If need be you can check the logs with:
```sh
docker compose logs -f app
```

To kill the container:
```sh
docker compose down
```


## License

This package is released under the [GNU General Public License version 3, 29 June 2007](https://www.gnu.org/licenses/gpl-3.0.html#license-text)

SPDX short identifier: [GPL-3.0-only](https://spdx.org/licenses/GPL-3.0-only.html)

[OSI link](https://opensource.org/license/gpl-3-0).


## Third Party Licenses

This package relies on many third party libraries. Please, carefully read the [NOTICE.md](./NOTICE.md) file.


## Going futher

Feel free to have a look at the [Wiki](../../wiki) if you want to go further.
