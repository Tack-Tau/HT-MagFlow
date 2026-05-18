#!/bin/bash
#
# Batch AFLOW prototype analysis for ternary magnet structure candidates.
#
# For each .vasp structure file:
#   1. Direct ternary match: run aflow --compare2prototypes on original structure
#   2. Binary match: merge T and T' species, run aflow --compare2prototypes --ignore_symmetry
#   3. If neither matches, mark as new prototype with _new suffix
#
# Output: candidates_w_proto.csv with columns:
#   structure_id,mattersim_e_hull,dft_e_hull,mattersim_energy_per_atom,vasp_energy_per_atom,spg_num,aflow_proto,pearson_symbol
#
# Usage:
#   ./run_aflow_prototype.sh --struct-dir ter_mag_results/struct --input ter_mag_results/candidates.csv
#   ./run_aflow_prototype.sh --struct-dir ter_mag_results/struct --input ter_mag_results/candidates.csv --np 4
#
# Requirements: aflow binary in PATH, python3

set -euo pipefail

STRUCT_DIR="ter_mag_results/struct"
INPUT_CSV="ter_mag_results/candidates.csv"
OUTPUT_CSV="ter_mag_results/aflow_prototypes.csv"
MERGED_CSV="ter_mag_results/candidates_w_proto.csv"
NP=1
AFLOW_NP=1

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --struct-dir DIR    Directory containing .vasp files (default: ter_mag_results/struct)"
    echo "  --input FILE        Input candidates CSV (default: ter_mag_results/candidates.csv)"
    echo "  --output FILE       AFLOW output CSV (default: ter_mag_results/aflow_prototypes.csv)"
    echo "  --merged FILE       Merged output CSV (default: ter_mag_results/candidates_w_proto.csv)"
    echo "  --np N              Number of parallel Python workers (default: 1)"
    echo "  --aflow-np N        Number of threads per aflow call (default: 1)"
    echo "  -h, --help          Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --struct-dir) STRUCT_DIR="$2"; shift 2 ;;
        --input) INPUT_CSV="$2"; shift 2 ;;
        --output) OUTPUT_CSV="$2"; shift 2 ;;
        --merged) MERGED_CSV="$2"; shift 2 ;;
        --np) NP="$2"; shift 2 ;;
        --aflow-np) AFLOW_NP="$2"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

if ! command -v aflow &>/dev/null; then
    echo "ERROR: aflow not found in PATH"
    exit 1
fi

if [[ ! -d "$STRUCT_DIR" ]]; then
    echo "ERROR: Structure directory not found: $STRUCT_DIR"
    exit 1
fi

if [[ ! -f "$INPUT_CSV" ]]; then
    echo "ERROR: Input CSV not found: $INPUT_CSV"
    exit 1
fi

echo "======================================================================"
echo "AFLOW Prototype Analysis (ternary + binary matching)"
echo "======================================================================"
echo "  aflow version: $(aflow --version 2>&1 | grep 'VERSION' | awk '{print $3}')"
echo "  Structure dir: $STRUCT_DIR"
echo "  Input CSV:     $INPUT_CSV"
echo "  Output:        $OUTPUT_CSV"
echo "  Merged:        $MERGED_CSV"
echo "  Parallel:      $NP"
echo "  AFLOW threads: $AFLOW_NP"
echo "======================================================================"
echo ""

# Run the Python helper which does all the heavy lifting
python3 "$(dirname "$0")/aflow_proto_helper.py" \
    --struct-dir "$STRUCT_DIR" \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_CSV" \
    --merged "$MERGED_CSV" \
    --np "$NP" \
    --aflow-np "$AFLOW_NP"
