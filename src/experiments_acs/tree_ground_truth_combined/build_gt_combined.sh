#!/usr/bin/env bash
set -e

# ---------------- Configuration ----------------
# todo: change
DISTR_CLUSTER=true
DISTR_TIMESTAMP="2026-04-16__18-20-00"
DISTR_EXPERIMENT_NAME="ground_truth_income"
# ----------------------------------------------
BINNED_CLUSTER=true
BINNED_TIMESTAMP="2026-04-17__15-30-00"
BINNED_EXPERIMENT_NAME="ground_truth_few_bins"
# ----------------------------------------------

# Paths
EXPERIMENT_NAME="gt_combined_tree"
TIMESTAMP=$(date +"%Y-%m-%d__%H-%M-%S")
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
BASE_PATH="$PROJECT_ROOT/results/$EXPERIMENT_NAME/$TIMESTAMP"
TEX_OUT="$BASE_PATH/main.tex"

# Parameters
SUBSAMPLING_FACTOR=5
LOWER_LIMIT=-20000
UPPER_LIMIT=250000

######################################################################

# Generate LaTeX source for variance tree
echo "→ Generating LaTeX for variance tree"
python3 gt_combined_tree.py "$DISTR_EXPERIMENT_NAME" "$DISTR_TIMESTAMP" "$DISTR_CLUSTER" \
  "$BINNED_EXPERIMENT_NAME" "$BINNED_TIMESTAMP" "$BINNED_CLUSTER" \
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