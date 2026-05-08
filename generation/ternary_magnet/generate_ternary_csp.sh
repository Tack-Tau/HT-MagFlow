#!/bin/bash
#SBATCH --job-name=gen_ter_mag_csp
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/gen_ter_mag_csp_%j.out
#SBATCH --error=logs/gen_ter_mag_csp_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=stao@uncc.edu

# Load modules
module purge
module load cuda/11.8

# Activate conda environment
source ~/.bashrc
conda activate mattergen

# Navigate to working directory
cd $HOME/SOFT/mattergen_test/ternary_magnet

# Create logs directory if it doesn't exist
mkdir -p logs

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Generation parameters
CSP_MODEL="$HOME/SOFT/mattergen_test/outputs/singlerun/2025-10-16/18-55-08"
COMPOSITIONS_FILE="ternary_magnet_compositions.json"
OUTPUT_DIR="../results/ternary_csp_magnets"
STRUCTURES_PER_ATOM=2.0
MAX_BATCH_SIZE=100  # Maximum structures per GPU batch (splits larger requests to avoid OOM)
TIMEOUT_PER_BATCH=1800  # Base timeout in seconds per batch (30 minutes) - auto-scaled for multiple batches
MAX_COMPOSITIONS=-1
START_INDEX=0

echo "=========================================="
echo "Ternary Magnet Generation (CSP Mode)"
echo "=========================================="
echo "CSP Model: $CSP_MODEL"
echo "Compositions: $COMPOSITIONS_FILE"
echo "Output: $OUTPUT_DIR"
echo "Structures per atom: $STRUCTURES_PER_ATOM"
echo "Max batch size: $MAX_BATCH_SIZE (auto-split for large requests)"
echo "Max compositions: $MAX_COMPOSITIONS"
echo "=========================================="

# Check if CSP model exists
if [ ! -d "$CSP_MODEL" ]; then
    echo "ERROR: CSP model directory not found: $CSP_MODEL"
    echo ""
    echo "Please update CSP_MODEL path in this script with your fine-tuned CSP checkpoint"
    echo "Example: outputs/singlerun/2025-10-16/18-55-08"
    exit 1
fi

# Check if compositions file exists
if [ ! -f "$COMPOSITIONS_FILE" ]; then
    echo "ERROR: Compositions file not found: $COMPOSITIONS_FILE"
    echo ""
    echo "Please run search_ternary_magnets.py first to generate compositions"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run generation
# The script will skip compositions that already have generated structures
# This allows you to resubmit the job if it times out
python generate_structures_batch.py \
    --compositions "$COMPOSITIONS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --model "$CSP_MODEL" \
    --structures-per-atom $STRUCTURES_PER_ATOM \
    --max-batch-size $MAX_BATCH_SIZE \
    --timeout $TIMEOUT_PER_BATCH \
    --max-compositions $MAX_COMPOSITIONS \
    --start-index $START_INDEX

echo ""
echo " Structure generation completed"

# ============================================================
# STEP 2: Create Summary
# ============================================================
echo ""
echo "STEP 2: Creating summary..."
echo "------------------------------------------------------------"

# Call the Python script that creates summary statistics
python summarize_generation.py \
    --output-dir "$OUTPUT_DIR"

SUMMARY_STATUS=$?

if [ $SUMMARY_STATUS -ne 0 ]; then
    echo ""
    echo "WARNING: Summary generation failed with exit code $SUMMARY_STATUS"
    # Don't exit, summary is not critical
fi

echo ""
echo " Summary created"

# ============================================================
# COMPLETION
# ============================================================
echo ""
echo "=========================================="
echo "Ternary magnet generation completed!"
echo "=========================================="
echo ""
echo "Results location: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
echo "  • Structure directories: ${OUTPUT_DIR}/*_structures/"
echo "  • Generation statistics: ${OUTPUT_DIR}/generation_statistics.json"
echo "  • Generation summary: ${OUTPUT_DIR}/generation_summary.json"
if [ -f "${OUTPUT_DIR}/failed_compositions.txt" ]; then
    echo "  • Failed compositions: ${OUTPUT_DIR}/failed_compositions.txt"
fi
echo ""
echo "Next steps:"
echo "  1. Review generation_summary.json"
echo "  2. Evaluate structures: sbatch ../mattergen_evaluate.sh"
echo "  3. Filter by stability"
echo ""
echo "=========================================="

