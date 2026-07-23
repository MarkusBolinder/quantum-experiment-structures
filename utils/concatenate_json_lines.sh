#!/usr/bin/env bash

set -euo pipefail

# Usage:  sh concatenate_json_lines.sh <target_directory> <max_rows_per_file>

# validate command line arguments
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 <target_directory> <max_rows_per_file>"
    exit 1
fi

TARGET_DIR="$1"
MAX_ROWS="$2"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

if ! [[ "$MAX_ROWS" =~ ^[0-9]+$ ]] || [ "$MAX_ROWS" -le 0 ]; then
    echo "Error: <max_rows_per_file> must be a positive integer."
    exit 1
fi

echo "--- Processing JSONL files in: $TARGET_DIR ---"
echo "Max rows per output file: $MAX_ROWS"

# find all .jsonl files in subdirectories (w/o matching top-level files)
mapfile -d '' JSONL_FILES < <(find "$TARGET_DIR" -mindepth 2 -type f -name "*.jsonl" -print0 | sort -z)

TOTAL_FILES=${#JSONL_FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No .jsonl files found in subdirectories under '$TARGET_DIR'."
    exit 0
fi

echo "Found $TOTAL_FILES .jsonl files across subdirectories."

# calculate total line count via wc -l
echo "Checking line counts..."
TOTAL_LINES=$(wc -l "${JSONL_FILES[@]}" | tail -n 1 | awk '{print $1}')

# calculate output file count (ceiling division)
EXPECTED_OUTPUT_FILES=$(((TOTAL_LINES + MAX_ROWS - 1) / MAX_ROWS))

echo "Total rows across all files: $TOTAL_LINES"
echo "Expected output files to generate: $EXPECTED_OUTPUT_FILES"

# concatenate and split output deterministically
echo "Combining and chunking files..."

cat "${JSONL_FILES[@]}" | split \
    -l "$MAX_ROWS" \
    --numeric-suffixes=0 \
    -a 5 \
    --additional-suffix=.jsonl \
    - \
    "${TARGET_DIR}/part_"

echo "---"
echo "Done! Generated files saved directly in '${TARGET_DIR}/':"
ls -1 "${TARGET_DIR}"/part_*.jsonl | head -n 5
if [ "$EXPECTED_OUTPUT_FILES" -gt 5 ]; then
    echo "... and $((EXPECTED_OUTPUT_FILES - 5)) more file(s)."
fi
