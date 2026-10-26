#!/usr/bin/env bash
# Copy the experiment folders into one flat working directory, so that every
# script finds its imports and data files by bare name, exactly as when it ran.
# Usage: tools/flatten_for_rerun.sh OUT_DIR    then: cd OUT_DIR && python3 exp93_run.py
set -euo pipefail
out="${1:?usage: $0 OUT_DIR}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out"
for d in archive/01_parser_pipeline archive/02_discocat_paper archive/03_semantic_geometry \
         archive/04_frames_scaling_hardware archive/06_trained_blocks \
         archive/07_capacity_levers archive/08_tokenisation arabic_axis; do
  cp -R "$repo/$d/." "$out/"
done
echo "flattened into $out"
