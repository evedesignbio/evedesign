#!/bin/sh
# Builds the two test images from two environments with different numpy/pandas
# versions (each with evedesign's dependencies and the remote extra installed),
# installing this checkout of evedesign into copies of them.
#
#   tests/remote/docker/build_test_images.sh <envA> <envB>
set -e
here=$(cd "$(dirname "$0")" && pwd)
checkout=$(cd "$here/../../.." && pwd)
for pair in "isoelectric:$1" "molweight:$2"; do
  name=${pair%%:*}; env=${pair#*:}
  ctx=$(mktemp -d)
  trap 'rm -rf "$ctx"' EXIT   # also on a failed build
  cp -a "$env" "$ctx/env"
  VIRTUAL_ENV="$ctx/env" uv pip install -q --no-deps --reinstall-package evedesign "$checkout"
  # model images hold models, not the pipeline (and an editable install would not resolve)
  VIRTUAL_ENV="$ctx/env" uv pip uninstall -q evedesign-server 2>/dev/null || true
  cp "$here/Dockerfile" "$ctx/"; cp "$here/card_$name.md" "$ctx/README.md"
  docker build -q -t "evedesign-test-$name:latest" "$ctx"
  rm -rf "$ctx"
done
