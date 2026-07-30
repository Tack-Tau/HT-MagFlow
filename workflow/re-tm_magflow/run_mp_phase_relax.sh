#!/bin/bash
# MP Phase Relaxation workflow manager - Submits manager as SLURM job
# Usage: bash run_mp_phase_relax.sh [options]

set -e

# Default values
CACHE_FILE="./mp_RE-TM_phase_cache.json"
OUTPUT_DIR="/scratch/$USER/mp_phase_relax"
MAX_CONCURRENT=10
CHECK_INTERVAL=60
DB_NAME="mp_RE-TM_phase_flow.json"
CONDA_ENV="vaspflow"
RETRY_FAILED=false

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================================================"
echo "MP Phase Relaxation - Missing Prototype VASP Workflow"
echo "========================================================================"
echo ""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cache)
            CACHE_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --check-interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --retry-failed)
            RETRY_FAILED=true
            shift
            ;;
        --help)
            echo "Usage: bash run_mp_phase_relax.sh [options]"
            echo ""
            echo "Submits the MP phase relaxation manager as a SLURM job."
            echo "Relaxes missing prototype-lanthanide combinations from the MP cache."
            echo ""
            echo "Options:"
            echo "  --cache FILE               MP phase cache JSON (default: ./mp_RE-TM_phase_cache.json)"
            echo "  --output-dir DIR           VASP job output directory (default: /scratch/\$USER/mp_phase_relax)"
            echo "  --max-concurrent N         Max concurrent VASP jobs (default: 10)"
            echo "  --check-interval SECONDS   Status check interval (default: 60)"
            echo "  --conda-env NAME           Conda environment name (default: vaspflow)"
            echo "  --retry-failed             Retry FAILED jobs using CONTCAR as starting geometry"
            echo ""
            echo "Example:"
            echo "  bash run_mp_phase_relax.sh --max-concurrent 20"
            echo "  bash run_mp_phase_relax.sh --retry-failed --max-concurrent 20"
            echo ""
            echo "Monitoring:"
            echo "  View log:        tail -f mp_phase_relax_<JOBID>.out"
            echo "  Check queue:     squeue -u \$USER"
            echo "  Cancel:          scancel <JOBID>"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Expand paths
CACHE_FILE=$(eval echo "$CACHE_FILE")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")

# Check if cache file exists (not needed for --retry-failed)
if [ "$RETRY_FAILED" != "true" ] && [ ! -f "$CACHE_FILE" ]; then
    echo -e "${RED}Error: Cache file not found: $CACHE_FILE${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if database exists
DB_PATH="$OUTPUT_DIR/$DB_NAME"
if [ -f "$DB_PATH" ]; then
    echo -e "${YELLOW}Database already exists: $DB_PATH${NC}"
    echo "The workflow will resume from previous state."
    echo ""
fi

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Output dir: $OUTPUT_DIR"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Database: $DB_PATH"
echo "  Conda env: $CONDA_ENV"
if [ "$RETRY_FAILED" = "true" ]; then
    echo "  Mode: RETRY FAILED jobs"
else
    echo "  Cache file: $CACHE_FILE"
fi
echo ""

# Export variables for SLURM script
export CACHE_FILE
export OUTPUT_DIR
export MAX_CONCURRENT
export CHECK_INTERVAL
export DB_NAME
export CONDA_ENV
export RETRY_FAILED

# Submit the workflow manager job
echo "Submitting MP phase relaxation manager as SLURM job..."
JOBID=$(sbatch submit_mp_phase_relax.sh | awk '{print $NF}')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Workflow manager submitted successfully!${NC}"
    echo ""
    echo "Job ID: $JOBID"
    echo ""
    echo "Monitoring commands:"
    echo "  View log:        tail -f mp_phase_relax_${JOBID}.out"
    echo "  Check status:    squeue -j $JOBID"
    echo "  Cancel job:      scancel $JOBID"
    echo ""
else
    echo -e "${RED}Error: Failed to submit job${NC}"
    exit 1
fi
