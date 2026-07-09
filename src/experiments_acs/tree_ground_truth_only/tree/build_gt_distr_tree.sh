#!/usr/bin/env bash
set -e

# ---------------- Configuration ----------------
# todo: change
GT_CLUSTER=true
GT_TIMESTAMP="2026-04-16__18-14-19"
GT_EXPERIMENT_NAME="ground_truth_income"
# ----------------------------------------------

# Paths
EXPERIMENT_NAME="gt_income_distribution_tree"
TIMESTAMP=$(date +"%Y-%m-%d__%H-%M-%S")
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
BASE_PATH="$PROJECT_ROOT/results/$EXPERIMENT_NAME/$TIMESTAMP"
TEX_OUT="$BASE_PATH/main.tex"

# Parameters
SUBSAMPLING_FACTOR=4
LOWER_LIMIT=-20000
UPPER_LIMIT=250000

######################################################################

# Generate LaTeX source for variance tree
echo "→ Generating LaTeX for variance tree"
python3 gt_income_tree.py "$GT_EXPERIMENT_NAME" "$GT_TIMESTAMP" "$GT_CLUSTER" \
  "$SUBSAMPLING_FACTOR" "$LOWER_LIMIT" "$UPPER_LIMIT"

# Delay (wait for plots to generate)
echo "→ Waiting for plots to generate..."
sleep 2

# Compile source for binary tree
echo "→ Compiling PDF for variance tree"
pdflatex \
  -interaction=nonstopmode \
  -output-directory="$BASE_PATH" \
  "$TEX_OUT" > /dev/null

echo "✓ Generated PDF for binary tree"
echo "**************************************"

######################################################################

# Plot output directory
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
PLOT_DIR="$PROJECT_ROOT/plots/$EXPERIMENT_NAME"
PLOT_PATH="$PLOT_DIR/$TIMESTAMP"
mkdir -p "$PLOT_DIR"

# Copy PDFs to output directory
cp "$BASE_PATH/main.pdf" "$PLOT_PATH.pdf"

echo "✓ Copied PDFs to $PLOT_PATH"

######################################################################