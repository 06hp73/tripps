#!/usr/bin/env bash
# Build tripps.app — the thing a person double-clicks.
#
# A .app is a directory with a particular shape, so this needs no Xcode: `sips` and
# `iconutil` are ordinary macOS binaries, not Command Line Tools shims. Run it on a Mac;
# the result is a bundle that launches run.sh from wherever the folder happens to be.
set -euo pipefail
cd "$(dirname "$0")"

APP="tripps.app"
say() { printf '\033[33m▸\033[0m %s\n' "$1"; }

[ "$(uname -s)" = "Darwin" ] || { echo "tripps.app is a macOS bundle; build it on a Mac." >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- Info.plist -------------------------------------------------------------
# LSBackgroundOnly is deliberately absent: the launcher opens a browser, and a background-only
# process cannot bring one to the front on first launch.
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>tripps</string>
  <key>CFBundleDisplayName</key>       <string>tripps</string>
  <key>CFBundleIdentifier</key>        <string>se.tripps.launcher</string>
  <key>CFBundleVersion</key>           <string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleExecutable</key>        <string>tripps</string>
  <key>CFBundleIconFile</key>          <string>tripps</string>
  <key>LSMinimumSystemVersion</key>    <string>11.0</string>
  <key>NSHighResolutionCapable</key>   <true/>
</dict>
</plist>
PLIST

# --- launcher ---------------------------------------------------------------
# The bundle sits inside the project folder, so the project is three levels up from the
# executable. Terminal is not involved: output goes to the system log, and anything a person
# needs to see is on the boot screen in their browser.
cat > "$APP/Contents/MacOS/tripps" <<'LAUNCH'
#!/bin/bash
here="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$here" || exit 1
exec ./run.sh
LAUNCH
chmod +x "$APP/Contents/MacOS/tripps"

# --- icon -------------------------------------------------------------------
# The departure board itself, at icon scale: an amber rule under a few rows, one of them the
# green that means a leg costs nothing.
ICON_SVG="$(mktemp -t trippsicon).svg"
cat > "$ICON_SVG" <<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <rect width="1024" height="1024" rx="228" fill="#0E1116"/>
  <rect x="150" y="742" width="724" height="26" fill="#F5C518"/>
  <rect x="150" y="250" width="250" height="74" rx="6" fill="#E8E6E1"/>
  <rect x="440" y="250" width="434" height="74" rx="6" fill="#F5C518"/>
  <rect x="150" y="400" width="420" height="74" rx="6" fill="#8A93A0"/>
  <rect x="610" y="400" width="264" height="74" rx="6" fill="#3DD68C"/>
  <rect x="150" y="550" width="316" height="74" rx="6" fill="#8A93A0"/>
  <rect x="506" y="550" width="368" height="74" rx="6" fill="#5C6672"/>
</svg>
SVG

ICONSET="$(mktemp -d)/tripps.iconset"
mkdir -p "$ICONSET"
for sz in 16 32 64 128 256 512 1024; do
  sips -s format png -z "$sz" "$sz" "$ICON_SVG" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null 2>&1
done
# .icns wants the @2x names too; reuse the larger renders rather than upscaling.
mv "$ICONSET/icon_32x32.png"     "$ICONSET/icon_16x16@2x.png"     2>/dev/null || true
cp "$ICONSET/icon_64x64.png"     "$ICONSET/icon_32x32@2x.png"     2>/dev/null || true
cp "$ICONSET/icon_256x256.png"   "$ICONSET/icon_128x128@2x.png"   2>/dev/null || true
cp "$ICONSET/icon_512x512.png"   "$ICONSET/icon_256x256@2x.png"   2>/dev/null || true
cp "$ICONSET/icon_1024x1024.png" "$ICONSET/icon_512x512@2x.png"   2>/dev/null || true
rm -f "$ICONSET/icon_64x64.png" "$ICONSET/icon_1024x1024.png"

if iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/tripps.icns" 2>/dev/null; then
  say "icon built"
else
  say "iconutil unavailable — shipping without a custom icon"
fi

# An ad-hoc signature keeps macOS from treating the bundle as damaged when it is moved.
# It is not notarization and does not remove the quarantine prompt on a downloaded ZIP.
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 && say "ad-hoc signed" || true

say "built $APP"
