#!/bin/bash

fname=$(basename "$1" .metal)
dirname=$(dirname "$1")
new_path="${dirname}/${fname}"
xcrun -sdk macosx metal -c $new_path.metal -o $new_path.air
xcrun -sdk macosx metallib $new_path.air -o $new_path.metallib
