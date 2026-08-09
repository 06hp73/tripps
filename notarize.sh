#!/usr/bin/env bash
# Sign, notarize and staple tripps.app, so a downloaded copy opens on the first double-click
# instead of "tripps cannot be opened because it is from an unidentified developer".
#
#   ./make-release.sh --with-feed
#   ./notarize.sh
#
# WHAT YOU NEED FIRST (once, and only you can do these — they involve your Apple account):
#
#   1. Apple Developer Program membership (99 USD/year).
#   2. A "Developer ID Application" certificate in your login keychain. Create it in
#      Xcode → Settings → Accounts → Manage Certificates, or on developer.apple.com.
#      Check it landed:  security find-identity -v -p codesigning
#   3. Credentials stored in the keychain, so they live in macOS and never in a script,
#      a file, or a shell history:
#
#        xcrun notarytool store-credentials tripps-notary \
#          --apple-id "you@example.com" --team-id "YOURTEAMID" \
#          --password "app-specific-password-from-appleid.apple.com"
#
#      Generate that app-specific password at appleid.apple.com → Sign-In and Security.
#      It is not your Apple ID password, and it can be revoked on its own.
#
# Override the defaults with TRIPPS_SIGN_IDENTITY and TRIPPS_NOTARY_PROFILE if you named
# things differently.
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="${TRIPPS_NOTARY_PROFILE:-tripps-notary}"
say()  { printf '\033[33m▸\033[0m %s\n' "$1"; }
die()  { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "notarization is a macOS process; run this on a Mac."
command -v xcrun >/dev/null 2>&1 || die "xcrun is missing. notarytool ships with Xcode or the
  Command Line Tools — this is the one step that needs them, on your build machine only.
  Nobody downloading the result needs anything."

# --- the signing identity ----------------------------------------------------
IDENTITY="${TRIPPS_SIGN_IDENTITY:-}"
if [ -z "$IDENTITY" ]; then
  # `|| true`: with `set -o pipefail`, grep finding nothing fails the whole pipeline, and
  # inside a command substitution that kills the script under `set -e` — silently, before
  # the message below can explain that the certificate is simply missing.
  IDENTITY="$( { security find-identity -v -p codesigning 2>/dev/null \
                 | grep "Developer ID Application" | head -1 \
                 | sed -E 's/.*"(.*)"/\1/'; } || true )"
fi
[ -n "$IDENTITY" ] || die "no \"Developer ID Application\" certificate found.
  Apple will only notarize software signed with one; an ad-hoc signature cannot be
  notarized, which is why the plain build still shows the quarantine prompt.
  See the header of this script for how to get one."
say "signing as: $IDENTITY"

xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1 \
  || die "no stored credentials under the keychain profile \"$PROFILE\".
  Run the notarytool store-credentials command in this script's header first."

# --- find what we are shipping ----------------------------------------------
ARCH_NAME="$(uname -m)"; [ "$ARCH_NAME" = "arm64" ] || ARCH_NAME=x86_64
OUT="dist/tripps-macos-${ARCH_NAME}"
APP="$OUT/tripps.app"
[ -d "$APP" ] || die "no $APP. Build it first:  ./make-release.sh --with-feed"

# --- sign, inside out --------------------------------------------------------
# Every Mach-O inside the bundle must carry the same signature, and nested code has to be
# signed before the thing that contains it — a signature records what it wraps, so signing
# the outside first would seal a hash that the inner signing then invalidates. `--deep` is
# not a substitute: Apple documents it as unreliable for exactly this.
say "signing nested binaries (this takes a moment)"
COUNT=0
while IFS= read -r f; do
  case "$(file -b "$f" 2>/dev/null)" in
    *Mach-O*)
      codesign --force --timestamp --options runtime --sign "$IDENTITY" "$f" >/dev/null 2>&1 \
        && COUNT=$((COUNT+1))
      ;;
  esac
done < <(find "$APP/Contents/Resources" -type f)
say "signed $COUNT nested binaries"

# The hardened runtime blocks unsigned code from loading, and CPython legitimately maps
# writable/executable pages and loads .so files at runtime. These entitlements permit that
# and nothing else.
ENT="$(mktemp -t trippsent).plist"
cat > "$ENT" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.cs.allow-jit</key>                            <true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key>     <true/>
  <key>com.apple.security.cs.disable-library-validation</key>           <true/>
</dict>
</plist>
PLIST

say "signing the bundle"
codesign --force --timestamp --options runtime --entitlements "$ENT" \
         --sign "$IDENTITY" "$APP"
codesign --verify --strict --verbose=2 "$APP" 2>&1 | tail -2

# --- submit ------------------------------------------------------------------
# Apple wants a ZIP for submission; ditto is the archiver that preserves the bundle's
# symlinks and extended attributes intact.
SUB="dist/tripps-notarize-${ARCH_NAME}.zip"
rm -f "$SUB"
/usr/bin/ditto -c -k --keepParent "$APP" "$SUB"

say "submitting to Apple (usually a few minutes)"
xcrun notarytool submit "$SUB" --keychain-profile "$PROFILE" --wait \
  || die "notarization failed. For the reasons:
    xcrun notarytool log <submission-id> --keychain-profile $PROFILE"

# --- staple ------------------------------------------------------------------
# Stapling attaches the ticket to the app so it verifies even offline. A ticket cannot be
# stapled to a .zip, so the app is stapled and then re-zipped for distribution.
say "stapling the ticket"
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"

say "verifying the way Gatekeeper will"
spctl --assess --type execute --verbose=4 "$APP" 2>&1 | tail -3

ZIP="dist/tripps-macos-${ARCH_NAME}.zip"
rm -f "$ZIP" "$SUB"
( cd dist && zip -qry "$(basename "$ZIP")" "$(basename "$OUT")" )
du -sh "$ZIP" | awk '{print "  " $2 "  " $1}'
say "done — this build opens on the first double-click, with no warning"
