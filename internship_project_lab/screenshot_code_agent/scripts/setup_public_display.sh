#!/usr/bin/env bash
# Project-local Xvfb binary. No sudo installation or shared environment changes.
set -euo pipefail
SCA_DISPLAY_ROOT=/data/zjr/screenshot_code_agent/workflow_v3_runtime
mkdir -p "$SCA_DISPLAY_ROOT/xvfb_download" "$SCA_DISPLAY_ROOT/xvfb_extract"
if [[ ! -x "$SCA_DISPLAY_ROOT/xvfb_extract/usr/bin/Xvfb" ]]; then
  cd "$SCA_DISPLAY_ROOT/xvfb_download"
  apt-get download xvfb xserver-common x11-xkb-utils libxfont2 libunwind8
  for SCA_DISPLAY_DEB in ./*.deb; do
    dpkg-deb -x "$SCA_DISPLAY_DEB" "$SCA_DISPLAY_ROOT/xvfb_extract"
  done
fi
command -v xkbcomp >/dev/null
if ldd "$SCA_DISPLAY_ROOT/xvfb_extract/usr/bin/Xvfb" | rg 'not found'; then
  echo 'Missing host Xvfb dependencies; inspect ldd output before running.' >&2
  exit 1
fi
echo "$SCA_DISPLAY_ROOT/xvfb_extract/usr/bin/Xvfb"
