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
#
# macOS ships Python 3.9, which is too old, and `/usr/bin/python3` is a stub that pops the
# Xcode Command Line Tools installer when no toolchain is present — as do `git`, `clang` and
# `swift`. `curl`, `tar` and `openssl` are real binaries on every Mac. So when there is no
# usable Python we fetch our own with those three and nothing else: no Xcode, no Homebrew,
# no admin password.
#
# PYTHON_PIN: bump these four together. The checksums were taken from the published assets
# at pin time (the project does not publish .sha256 siblings, so they are recorded here).
PY_TAG=20260807
PY_VER=3.12.13
PY_DIR=".python"

py_asset() {                       # -> "<arch-triple> <sha256>"
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64)  echo "aarch64-apple-darwin 4201588fc5051c2ba988abbe1f033d318965ee378fadf7fb7ef79882ba7be84b" ;;
    Darwin/x86_64) echo "x86_64-apple-darwin ce9dc826a3215d5deadf6d7ba409a882b8d431192c4c06deb34ff00f93ceb4f5" ;;
    Linux/x86_64)  echo "x86_64-unknown-linux-gnu 5bd6f36fd7ef02b909234c94dca9994ef0da06ace3bc3cece4fe27870e9cdbbe" ;;
    Linux/aarch64) echo "aarch64-unknown-linux-gnu e2a33a26bae0f0975a9786c2e3beaee9cfeb35f856bdd273ff10ae35cf7e06ce" ;;
    *) return 1 ;;
  esac
}

sha256_of() {
  if command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$1" | sed 's/.*= *//'
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

find_python() {
  # Ours first: once fetched it is the known-good one, and it never triggers an installer.
  if [ -x "$PY_DIR/bin/python3" ]; then echo "$PY_DIR/bin/python3"; return 0; fi
  for c in python3.14 python3.13 python3.12 python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null \
      && { command -v "$c"; return 0; }
  done
  return 1
}

fetch_python() {
  read -r arch sum <<EOF
$(py_asset)
EOF
  [ -n "${arch:-}" ] || die "no prebuilt Python for $(uname -s)/$(uname -m).
  Install Python 3.12 or newer yourself, then run this again."

  local url="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_TAG}/cpython-${PY_VER}%2B${PY_TAG}-${arch}-install_only.tar.gz"
  local tmp="$PY_DIR.part"
  rm -rf "$tmp"; mkdir -p "$tmp"

  say "fetching Python ${PY_VER} (~25 MB, one time — no Xcode or Homebrew needed)"
  curl -fL --retry 3 --progress-bar -o "$tmp/py.tar.gz" "$url" \
    || die "could not download Python. Check the network and try again."

  local got; got="$(sha256_of "$tmp/py.tar.gz")"
  [ "$got" = "$sum" ] || { rm -rf "$tmp"; die "the downloaded Python did not match its
  expected checksum, so it was discarded.
    expected $sum
    got      $got"; }

  tar -xzf "$tmp/py.tar.gz" -C "$tmp" || { rm -rf "$tmp"; die "could not unpack Python."; }
  rm -f "$tmp/py.tar.gz"
  # The archive unpacks to a single `python/` directory; move it into place atomically so an
  # interrupted run never leaves a half-extracted interpreter that looks usable.
  rm -rf "$PY_DIR"
  mv "$tmp/python" "$PY_DIR"
  rm -rf "$tmp"
}

# --- 2. the virtualenv ------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    say "creating $VENV (uv)"
    uv venv --python 3.12 "$VENV" >/dev/null
  else
    PY="$(find_python)" || { fetch_python; PY="$PY_DIR/bin/python3"; }
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
