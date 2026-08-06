#!/usr/bin/env bash
# One press: set up whatever is missing, then start tripps.
#
#   ./run.sh                    web UI on http://127.0.0.1:8000
#   ./run.sh search "Lund C" "Stockholm C" --date 2026-08-09
#   ./run.sh cards providers
#
# Everything here is idempotent and skippable: a second run reuses the venv, the install and
# the 65 MB feed, so it starts in about a second. Nothing needs a key or an account.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${TRIPPS_PORT:-8000}"
VENV=".venv"
STAMP="$VENV/.tripps-install-stamp"

say() { printf '\033[33m▸\033[0m %s\n' "$1"; }
die() { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

# --- 1. an interpreter new enough to run this -------------------------------
find_python() {
  for c in python3.14 python3.13 python3.12 python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null \
      && { echo "$c"; return 0; }
  done
  return 1
}

# --- 2. the virtualenv ------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    say "creating $VENV (uv)"
    uv venv --python 3.12 "$VENV" >/dev/null
  else
    PY="$(find_python)" || die "needs Python 3.12 or newer on PATH (tried python3.12 … python).
  macOS:  brew install python@3.12
  Debian: sudo apt install python3.12 python3.12-venv
  or install uv: https://docs.astral.sh/uv/"
    say "creating $VENV ($PY)"
    "$PY" -m venv "$VENV"
  fi
fi

# --- 3. dependencies, only when pyproject.toml is newer than the last install ---
if [ ! -f "$STAMP" ] || [ pyproject.toml -nt "$STAMP" ]; then
  say "installing dependencies (a minute the first time)"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV/bin/python" -q -e ".[dev]"
  else
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q -e ".[dev]"
  fi
  touch "$STAMP"
fi

# --- 4. the timetable feed --------------------------------------------------
# A feed that is present but unreadable is worse than none: without this check the run
# skips the download and fails deep in the parser. Downloads are atomic now, so this only
# catches files left by an older version or copied in by hand — but it costs milliseconds
# and turns "delete this file yourself" into something the button just handles.
FEED="${TRIPPS_GTFS_ZIP_PATH:-${TRIPPS_DATA_DIR:-data}/sweden.zip}"
if [ -f "$FEED" ] && ! "$VENV/bin/python" -c \
     'import sys, zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$FEED"; then
  say "the timetable feed is incomplete — discarding it and fetching again"
  rm -f "$FEED"
fi

if [ ! -f "$FEED" ]; then
  say "downloading the national timetable feed (~65 MB, CC0, no key needed)"
  "$VENV/bin/tripps" fetch-gtfs
fi

# --- 5. go ------------------------------------------------------------------
# Any argument means "you know what you want" — hand it straight to the CLI.
if [ $# -gt 0 ]; then
  exec "$VENV/bin/tripps" "$@"
fi

if "$VENV/bin/python" -c "
import socket, sys
s = socket.socket()
sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT)) == 0 else 1)
" 2>/dev/null; then
  die "port $PORT is already in use. Stop what is there, or: TRIPPS_PORT=8010 ./run.sh"
fi

URL="http://127.0.0.1:$PORT"
say "starting tripps on $URL  (ctrl-c to stop)"

# Open a browser once the server answers, without blocking the server itself. The first
# search parses the feed, so the page is up well before results are.
(
  for _ in $(seq 1 60); do
    if "$VENV/bin/python" -c "
import socket, sys
sys.exit(0 if socket.socket().connect_ex(('127.0.0.1', $PORT)) == 0 else 1)
" 2>/dev/null; then
      if command -v open >/dev/null 2>&1; then open "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
      fi
      exit 0
    fi
    sleep 0.5
  done
) &

exec "$VENV/bin/tripps" serve --port "$PORT"
