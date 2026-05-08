#!/bin/bash
#SBATCH --job-name=vaspflow_manager
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30-00:00:00
#SBATCH --output=workflow_manager_%j.out
#SBATCH --error=workflow_manager_%j.err

# VASPflow Workflow Manager - VASP calculations only
# Pre-screening should be done separately with prescreen.py

set -e

# Default parameters (can be overridden by environment variables)
RESULTS_DIR=${RESULTS_DIR:-"./mattergen_results/ternary_csp_electrides"}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/$USER/VASP_JOBS"}
MAX_CONCURRENT=${MAX_CONCURRENT:-10}
MAX_COMPOSITIONS=${MAX_COMPOSITIONS:-""}
MAX_STRUCTURES=${MAX_STRUCTURES:-5}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
CONDA_ENV=${CONDA_ENV:-"vaspflow"}
DB_NAME=${DB_NAME:-"workflow.json"}
PRESCREEN_RESULTS=${PRESCREEN_RESULTS:-"$OUTPUT_DIR/prescreening_stability.json"}

echo "========================================================================"
echo "VASPflow Workflow Manager (SLURM Job)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Load environment
source ~/.bashrc
conda activate $CONDA_ENV

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    exit 1
fi

echo "Conda environment: $CONDA_ENV"
echo "Python: $(which python3)"
echo ""

# Check pymatgen
if ! python3 -c "import pymatgen" 2>/dev/null; then
    echo "Error: pymatgen not found"
    exit 1
fi

# Expand paths
RESULTS_DIR=$(eval echo "$RESULTS_DIR")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")
PRESCREEN_RESULTS=$(eval echo "$PRESCREEN_RESULTS")

# Check results directory
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python3 magflow_manager.py"
CMD="$CMD --results-dir $RESULTS_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --max-concurrent $MAX_CONCURRENT"
CMD="$CMD --max-structures $MAX_STRUCTURES"
CMD="$CMD --check-interval $CHECK_INTERVAL"
CMD="$CMD --db $DB_NAME"

if [ -n "$MAX_COMPOSITIONS" ]; then
    CMD="$CMD --max-compositions $MAX_COMPOSITIONS"
fi

# Add pre-screening results if file exists
if [ -f "$PRESCREEN_RESULTS" ]; then
    CMD="$CMD --prescreen-results $PRESCREEN_RESULTS"
    echo "Pre-screening results found: $PRESCREEN_RESULTS"
else
    echo "Warning: Pre-screening results not found: $PRESCREEN_RESULTS"
    echo "Will process all structures without filtering"
fi

# Print configuration
echo "Configuration:"
echo "  Results dir: $RESULTS_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Max structures: $MAX_STRUCTURES"
echo "  Max compositions: ${MAX_COMPOSITIONS:-all}"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Database: $OUTPUT_DIR/$DB_NAME"
echo ""
echo "Note: Run prescreen.py first to generate prescreening_stability.json"
echo ""

# Check if resuming
if [ -f "$OUTPUT_DIR/$DB_NAME" ]; then
    echo "Database exists - resuming from previous state"
    echo ""
fi

echo "========================================================================"
echo "Starting workflow manager..."
echo "========================================================================"
echo ""
echo "Command: $CMD"
echo ""

# Run workflow manager
$CMD

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Workflow manager finished"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE

