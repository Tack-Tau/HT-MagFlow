#!/bin/bash
# Submit multiple parallel generation jobs for ternary magnets
# Each job processes a different subset of compositions

# Configuration
COMPOSITIONS_PER_JOB=400  # Adjust based on desired job size
COMPOSITIONS_FILE="ternary_magnet_compositions.json"
STRUCTURES_PER_ATOM=6.0  # Number of structures to generate per atom

# Auto-detect total compositions from JSON file
if [ ! -f "$COMPOSITIONS_FILE" ]; then
    echo "ERROR: Compositions file not found: $COMPOSITIONS_FILE"
    echo "Please run search_ternary_magnets.py first"
    exit 1
fi

# Count compositions in JSON file (works with python3 or jq)
if command -v python3 &> /dev/null; then
    TOTAL_COMPOSITIONS=$(python3 -c "import json; print(len(json.load(open('$COMPOSITIONS_FILE'))))")
elif command -v jq &> /dev/null; then
    TOTAL_COMPOSITIONS=$(jq 'length' "$COMPOSITIONS_FILE")
else
    echo "ERROR: Neither python3 nor jq found. Please install one of them."
    exit 1
fi

echo "Found $TOTAL_COMPOSITIONS compositions in $COMPOSITIONS_FILE"

# Calculate number of jobs needed
NUM_JOBS=$(( (TOTAL_COMPOSITIONS + COMPOSITIONS_PER_JOB - 1) / COMPOSITIONS_PER_JOB ))

echo "Submitting $NUM_JOBS parallel generation jobs"
echo "Each job will process $COMPOSITIONS_PER_JOB compositions"
echo "Structures per atom: $STRUCTURES_PER_ATOM"
echo "=========================================="

for i in $(seq 0 $((NUM_JOBS - 1))); do
    START_IDX=$((i * COMPOSITIONS_PER_JOB))
    
    # Create a temporary modified script for this batch
    TEMP_SCRIPT="generate_ternary_csp_batch${i}.sh"
    
    # Copy original script and modify parameters
    sed "s/^START_INDEX=.*/START_INDEX=$START_IDX/" generate_ternary_csp.sh | \
    sed "s/^MAX_COMPOSITIONS=.*/MAX_COMPOSITIONS=$COMPOSITIONS_PER_JOB/" | \
    sed "s/^STRUCTURES_PER_ATOM=.*/STRUCTURES_PER_ATOM=$STRUCTURES_PER_ATOM/" | \
    sed "s/gen_ter_mag_csp_%j/gen_ter_mag_csp_batch${i}_%j/" > $TEMP_SCRIPT
    
    # Submit the job
    JOB_ID=$(sbatch $TEMP_SCRIPT | awk '{print $NF}')
    
    echo "Submitted batch $i (compositions $START_IDX-$((START_IDX + COMPOSITIONS_PER_JOB - 1))): Job ID $JOB_ID"
    
    # Clean up temp script
    rm $TEMP_SCRIPT
    
    # Small delay to avoid overwhelming scheduler
    sleep 1
done

echo "=========================================="
echo "All jobs submitted!"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check progress: ls -d ../results/ternary_csp_magnets/*_structures | wc -l"
echo "=========================================="

