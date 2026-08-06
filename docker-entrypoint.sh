#!/bin/sh
# Fetch the national GTFS feed once if the data volume is empty, then serve.
# The feed is CC0 and the endpoint currently serves it unauthenticated — no key required.
set -e

DATA_DIR="${TRIPPS_DATA_DIR:-/app/data}"
FEED="${TRIPPS_GTFS_ZIP_PATH:-$DATA_DIR/sweden.zip}"

if [ ! -f "$FEED" ]; then
  echo "tripps: no GTFS feed at $FEED — fetching (~65 MB, no key required)..."
  if ! tripps fetch-gtfs; then
    echo "tripps: WARNING — fetch-gtfs failed. The server will start but searches need a feed." >&2
    echo "tripps: check outbound network access, or mount a data volume holding sweden.zip." >&2
  fi
fi

exec tripps serve --host 0.0.0.0 --port 8000
