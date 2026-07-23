#!/bin/bash
#SBATCH --job-name=mp_phase_relax
#SBATCH --partition=Apus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=30-00:00:00
#SBATCH --output=mp_phase_relax_%j.out
#SBATCH --error=mp_phase_relax_%j.err

# MP Phase Relaxation Manager - SLURM submission script

set -e

# Default parameters (overridden by environment variables from run_mp_phase_relax.sh)
CACHE_FILE=${CACHE_FILE:-"./mp_RE-TM_phase_cache.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/$USER/mp_phase_relax"}
MAX_CONCURRENT=${MAX_CONCURRENT:-10}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
CONDA_ENV=${CONDA_ENV:-"vaspflow"}
DB_NAME=${DB_NAME:-"mp_RE-TM_phase_flow.json"}

echo "========================================================================"
echo "MP Phase Relaxation Manager (SLURM Job)"
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
CACHE_FILE=$(eval echo "$CACHE_FILE")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")

# Check cache file
if [ ! -f "$CACHE_FILE" ]; then
    echo "Error: Cache file not found: $CACHE_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python3 mp_RE-TM_phase_relax.py"
CMD="$CMD --cache $CACHE_FILE"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --max-concurrent $MAX_CONCURRENT"
CMD="$CMD --check-interval $CHECK_INTERVAL"
CMD="$CMD --db $DB_NAME"

# Print configuration
echo "Configuration:"
echo "  Cache file: $CACHE_FILE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Database: $OUTPUT_DIR/$DB_NAME"
echo ""

# Check if resuming
if [ -f "$OUTPUT_DIR/$DB_NAME" ]; then
    echo "Database exists - resuming from previous state"
    echo ""
fi

echo "========================================================================"
echo "Starting MP phase relaxation manager..."
echo "========================================================================"
echo ""
echo "Command: $CMD"
echo ""

# Run workflow manager
$CMD

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "MP phase relaxation manager finished"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================================"

exit $EXIT_CODE
