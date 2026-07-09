#!/usr/bin/env bash
set -e

# ---------------- Configuration ----------------
# todo: change
GT_CLUSTER=true
GT_TIMESTAMP="2026-04-17__15-30-00"
GT_EXPERIMENT_NAME="ground_truth_few_bins"
GT_EQUAL_BAR_WIDTH=true
# ----------------------------------------------

# Paths
EXPERIMENT_NAME="ground_truth_few_bins_tree"
TIMESTAMP=$(date +"%Y-%m-%d__%H-%M-%S")
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
BASE_PATH="$PROJECT_ROOT/results/$EXPERIMENT_NAME/$TIMESTAMP"
TEX_OUT="$BASE_PATH/main.tex"

######################################################################

# Generate LaTeX source for variance tree
echo "→ Generating LaTeX for variance tree"
python3 gt_tree_few_bins.py "$GT_EXPERIMENT_NAME" "$GT_TIMESTAMP" "$GT_CLUSTER" "$GT_EQUAL_BAR_WIDTH"

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