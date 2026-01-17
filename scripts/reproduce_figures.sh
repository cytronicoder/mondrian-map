#!/bin/bash
# Reproduce Figures 1-2 from the Mondrian Map Paper
# Usage: ./scripts/reproduce_figures.sh [--no-cache]
#
# This script reproduces the GBM case study results using either:
# 1. Cached precomputed artifacts (fast, default)
# 2. Full pipeline from raw data (slow, requires PAGER API access)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
USE_CACHE="--use-cache"
if [[ "$1" == "--no-cache" ]]; then
    USE_CACHE="--no-cache"
    echo -e "${YELLOW}Running full pipeline without cache (this may take 10-30 minutes)${NC}"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not found${NC}"
    exit 1
fi

# Check for mondrian-map installation
if ! python3 -c "import mondrian_map" 2>/dev/null; then
    echo -e "${YELLOW}Installing mondrian-map package...${NC}"
    pip install -e .
fi

echo "======================================"
echo "Mondrian Map - Reproduce Paper Figures"
echo "======================================"
echo ""

# Create output directory
OUTPUT_DIR="outputs/gbm"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Step 1: Running GBM case study reproduction...${NC}"
echo ""

# Run the reproduction command
python3 -m mondrian_map.cli reproduce \
    --case-study gbm \
    --out "$OUTPUT_DIR" \
    $USE_CACHE

echo ""
echo -e "${GREEN}Step 2: Verifying outputs...${NC}"

# Check for expected output files
EXPECTED_FILES=(
    "$OUTPUT_DIR/entities.csv"
    "$OUTPUT_DIR/manifest.json"
)

MISSING_FILES=()
for file in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo -e "${RED}Warning: Some expected files are missing:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
else
    echo -e "${GREEN}All expected files generated successfully!${NC}"
fi

echo ""
echo "======================================"
echo "Reproduction Complete!"
echo "======================================"
echo ""
echo "Output files:"
echo "  - Entities: $OUTPUT_DIR/entities.csv"
echo "  - Manifest: $OUTPUT_DIR/manifest.json"

# Check for visualization
if [[ -f "$OUTPUT_DIR/mondrian_map.html" ]]; then
    echo "  - Visualization: $OUTPUT_DIR/mondrian_map.html"
    echo ""
    echo -e "${GREEN}Open the HTML file in a browser to view the interactive Mondrian Map${NC}"
fi

echo ""
echo "To regenerate individual visualizations:"
echo "  mondrian-map visualize --entities $OUTPUT_DIR/entities.csv --out figure.html --show-ids"
