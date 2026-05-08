#!/bin/bash
# VASPflow workflow manager - Submits workflow manager as SLURM job
# Usage: bash run_workflow.sh [options]

set -e

# Default values
RESULTS_DIR="./mattergen_results/ternary_csp_electrides"
OUTPUT_DIR="/scratch/$USER/VASP_JOBS"
MAX_CONCURRENT=10
MAX_COMPOSITIONS=""
MAX_STRUCTURES=0
CHECK_INTERVAL=60
DB_NAME="workflow.json"
CONDA_ENV="vaspflow"
PRESCREEN_RESULTS=""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================================================"
echo "VASPflow - High-Throughput VASP Workflow Manager"
echo "========================================================================"
echo ""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
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
        --max-compositions)
            MAX_COMPOSITIONS="$2"
            shift 2
            ;;
        --max-structures)
            MAX_STRUCTURES="$2"
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
        --prescreen-results)
            PRESCREEN_RESULTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash run_workflow.sh [options]"
            echo ""
            echo "This script submits the workflow manager as a SLURM job."
            echo "The workflow manager will run on a compute node and submit VASP jobs."
            echo ""
            echo "Options:"
            echo "  --results-dir DIR            MatterGen results directory"
            echo "  --output-dir DIR             VASP job output directory"
            echo "  --max-concurrent N           Max concurrent structures (default: 10)"
            echo "  --max-compositions N         Max compositions to process (default: all)"
            echo "  --max-structures N           Max structures per composition (default: 5, use 0 for all)"
            echo "  --check-interval SECONDS     Status check interval (default: 60)"
            echo "  --conda-env NAME             Conda environment name (default: vaspflow)"
            echo "  --prescreen-results FILE     Pre-screening results file (default: auto-detect)"
            echo ""
            echo "Example:"
            echo "  bash run_workflow.sh --max-concurrent 20 --max-structures 0"
            echo "  bash run_workflow.sh --prescreen-results ./VASP_JOBS/prescreening_stability.json"
            echo ""
            echo "Note: Run prescreen.py first to filter structures by stability."
            echo ""
            echo "Monitoring:"
            echo "  Check status:    python workflow_status.py ./VASP_JOBS/workflow.json"
            echo "  View log:        tail -f workflow_manager_<JOBID>.out"
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
RESULTS_DIR=$(eval echo "$RESULTS_DIR")
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")
if [ -n "$PRESCREEN_RESULTS" ]; then
    PRESCREEN_RESULTS=$(eval echo "$PRESCREEN_RESULTS")
else
    PRESCREEN_RESULTS="$OUTPUT_DIR/prescreening_stability.json"
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${RED}Error: Results directory not found: $RESULTS_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for pre-screening results
if [ -f "$PRESCREEN_RESULTS" ]; then
    PASSED_COUNT=$(jq '.summary.passed_prescreening' "$PRESCREEN_RESULTS" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}Pre-screening results found: $PRESCREEN_RESULTS${NC}"
    echo "  Structures passed: $PASSED_COUNT"
    echo ""
else
    echo -e "${YELLOW}Warning: Pre-screening results not found: $PRESCREEN_RESULTS${NC}"
    echo "  Workflow will process ALL structures without filtering."
    echo "  Recommend running: bash run_prescreen.sh first"
    echo ""
fi

# Check if database exists
DB_PATH="$OUTPUT_DIR/$DB_NAME"
if [ -f "$DB_PATH" ]; then
    echo -e "${YELLOW}Database already exists: $DB_PATH${NC}"
    echo "The workflow will resume from previous state."
    echo ""
fi

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Results dir: $RESULTS_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Max structures: $MAX_STRUCTURES"
if [ -n "$MAX_COMPOSITIONS" ]; then
    echo "  Max compositions: $MAX_COMPOSITIONS"
else
    echo "  Max compositions: all"
fi
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Database: $DB_PATH"
echo "  Conda env: $CONDA_ENV"
echo ""

# Export variables for SLURM script
export RESULTS_DIR
export OUTPUT_DIR
export MAX_CONCURRENT
export MAX_COMPOSITIONS
export MAX_STRUCTURES
export CHECK_INTERVAL
export DB_NAME
export CONDA_ENV
export PRESCREEN_RESULTS

# Submit the workflow manager job
echo "Submitting workflow manager as SLURM job..."
JOBID=$(sbatch submit_workflow_manager.sh | awk '{print $NF}')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Workflow manager submitted successfully!${NC}"
    echo ""
    echo "Job ID: $JOBID"
    echo ""
    echo "Monitoring commands:"
    echo "  View log:        tail -f workflow_manager_${JOBID}.out"
    echo "  Check status:    squeue -j $JOBID"
    echo "  Check workflow:  python workflow_status.py $OUTPUT_DIR/workflow.json"
    echo "  Cancel job:      scancel $JOBID"
    echo ""
    echo "The workflow manager will:"
    echo "  - Run on a compute node"
    echo "  - Submit and monitor VASP jobs automatically"
    echo "  - Run for up to 30 days"
    echo "  - Save progress to $DB_PATH"
    echo ""
else
    echo -e "${RED}Error: Failed to submit job${NC}"
    exit 1
fi

