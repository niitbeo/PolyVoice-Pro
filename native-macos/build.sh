#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
swiftc -O -parse-as-library \
  -target arm64-apple-macos13.0 \
  "$DIR/helper.swift" \
  -o "$DIR/voicerecore-macos-helper" \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker "$DIR/Info.plist"
echo "Built $DIR/voicerecore-macos-helper (target arm64-apple-macos13.0)"
