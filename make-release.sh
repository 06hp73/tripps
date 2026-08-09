#!/usr/bin/env bash
# Build the download: a folder that runs the moment it is unzipped.
#
#   ./make-release.sh              # for this Mac's architecture
#   ./make-release.sh --with-feed  # …and bake today's timetables in (+68 MB, opens offline)
#
# Everything is inside: a private Python, the dependencies, the source, the .app. No pip, no
# network and no prerequisites on the far end — which is the whole point, because the first
# thing a downloader hits otherwise is "install Python 3.12".
#
# Deps are installed into the interpreter's own site-packages rather than a virtualenv: a
# venv records absolute paths and would break the moment the folder is moved, and an editable
# install records them too. The app no longer depends on its own location (see
# config._default_data_dir), so a plain install is now correct as well as relocatable.
set -euo pipefail
cd "$(dirname "$0")"

WITH_FEED=0
[ "${1:-}" = "--with-feed" ] && WITH_FEED=1

say() { printf '\033[33m▸\033[0m %s\n' "$1"; }
die() { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "this packages the macOS bundle; run it on a Mac."

case "$(uname -m)" in
  arm64)  ARCH_NAME=arm64 ;;
  x86_64) ARCH_NAME=x86_64 ;;
  *) die "unknown architecture $(uname -m)" ;;
esac

OUT="dist/tripps-macos-${ARCH_NAME}"
ZIP="dist/tripps-macos-${ARCH_NAME}.zip"
rm -rf "$OUT" "$ZIP"; mkdir -p "$OUT"

# --- the private interpreter -------------------------------------------------
# run.sh already knows how to fetch and verify it; reuse that rather than duplicating the
# pin here, so there is exactly one place the version and checksums live.
TRIPPS_BOOTSTRAP_ONLY=1 ./run.sh
[ -x ".python/bin/python3" ] || die "the private Python was not produced; cannot package."

say "copying Python"
cp -R ".python" "$OUT/.python"

say "installing dependencies into it"
"$OUT/.python/bin/python3" -m pip install -q --upgrade pip >/dev/null
"$OUT/.python/bin/python3" -m pip install -q . >/dev/null

# --- the app ------------------------------------------------------------------
# Everything goes INSIDE the bundle. Gatekeeper only vouches for what a signature covers, so
# an interpreter sitting next to the .app rather than within it would leave the download
# quarantined however well the launcher itself were notarized — and the whole point of
# notarizing is to remove that prompt. One bundle, one signature, one verdict.
say "building tripps.app"
./build-app.sh >/dev/null
cp -R "tripps.app" "$OUT/tripps.app"

RES="$OUT/tripps.app/Contents/Resources"
mv "$OUT/.python" "$RES/python"
cp -R src "$RES/src"
cp pyproject.toml README.md LICENSE .env.example "$RES/"
cp run.sh "$RES/run.sh"

if [ "$WITH_FEED" = "1" ]; then
  [ -f data/sweden.zip ] || die "no data/sweden.zip to bake in. Run ./run.sh once first."
  say "baking in today's timetable feed"
  mkdir -p "$RES/data"
  cp data/sweden.zip "$RES/data/sweden.zip"
fi

cat > "$RES/.tripps-packaged" <<'NOTE'
This bundle ships its own Python with the dependencies already installed.
run.sh sees this file and uses it directly instead of creating a virtualenv.
NOTE

# A short note beside the app, for anyone who opens the folder before the app.
cat > "$OUT/README.txt" <<'NOTE'
tripps — the cheapest way across Sweden.

Double-click "tripps". Nothing to install first.

The first launch opens a browser showing what it is loading, then the planner.
Your searches, travel cards and downloaded timetables are kept in
~/Library/Application Support/tripps, so replacing this app never loses them.
NOTE

say "zipping"
mkdir -p dist
( cd dist && zip -qry "$(basename "$ZIP")" "$(basename "$OUT")" )
du -sh "$ZIP" | awk '{print "  " $2 "  " $1}'
say "done"
