#!/bin/sh
# Fetch the national GTFS feed once if the data volume is empty, then serve.
# The feed is CC0 but the download needs a (free) Trafiklab key in TRIPPS_TRAFIKLAB_GTFS_KEY.
set -e

DATA_DIR="${TRIPPS_DATA_DIR:-/app/data}"
FEED="${TRIPPS_GTFS_ZIP_PATH:-$DATA_DIR/sweden.zip}"

if [ ! -f "$FEED" ]; then
  echo "tripps: no GTFS feed at $FEED — fetching (needs TRIPPS_TRAFIKLAB_GTFS_KEY)..."
  if ! tripps fetch-gtfs; then
    echo "tripps: WARNING — fetch-gtfs failed. The server will start but searches need a feed." >&2
    echo "tripps: mount a data volume with sweden.zip, or set TRIPPS_TRAFIKLAB_GTFS_KEY." >&2
  fi
fi

exec tripps serve --host 0.0.0.0 --port 8000
