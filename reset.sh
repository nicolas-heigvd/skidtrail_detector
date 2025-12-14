#!/bin/bash

set -e

reset

docker compose down --remove-orphans

docker compose build --progress plain

docker compose up -d

docker compose logs --tail 100 -f
