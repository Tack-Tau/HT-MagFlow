#!/bin/bash
"""
Split prescreening_stability.json into separate files for Ce5Fe26 and Ce2Fe14B derivatives.
Updates summary statistics for each split file.

Usage:
    bash split_derivatives.sh [input_json] [output_dir]
    
Default:
    input_json = prescreening_stability.json
    output_dir = current directory
"""

# Parse arguments
INPUT_JSON="${1:-prescreening_stability.json}"
OUTPUT_DIR="${2:-.}"

# Check input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Input file not found: $INPUT_JSON"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Splitting Prescreening Results by Prototype"
echo "========================================================================"
echo "Input file: $INPUT_JSON"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Define patterns
# Ce5Fe26 pattern: A5B26 (from Ce10Fe52 prototype, reduced)
CE5FE26_PATTERN='^[A-Z][a-z]?5[A-Z][a-z]?26_'
CE5FE26_OUTPUT="$OUTPUT_DIR/prescreening_Ce5Fe26.json"

# Ce2Fe14B pattern: A2B14B (from Ce2Fe14B prototype)
CE2FE14B_PATTERN='^[A-Z][a-z]?2[A-Z][a-z]?14B_'
CE2FE14B_OUTPUT="$OUTPUT_DIR/prescreening_Ce2Fe14B.json"

echo "Processing Ce5Fe26 derivatives (from Ce10Fe52 prototype)..."
echo "  Pattern: $CE5FE26_PATTERN"
echo "  Output: $CE5FE26_OUTPUT"

# Extract Ce5Fe26 derivatives with updated summary
jq '{
  summary: {
    total_structures: ([.results[] | select(.structure_id | test("'"$CE5FE26_PATTERN"'"))] | length),
    passed_prescreening: ([.results[] | select(.structure_id | test("'"$CE5FE26_PATTERN"'") and .passed_prescreening == true)] | length),
    failed_prescreening: ([.results[] | select(.structure_id | test("'"$CE5FE26_PATTERN"'") and .passed_prescreening == false)] | length),
    hull_threshold: .summary.hull_threshold,
    energy_reference: .summary.energy_reference,
    prototype: "Ce10Fe52 (reduced to A5B26)",
    filtered_from: "'"$INPUT_JSON"'"
  },
  results: [.results[] | select(.structure_id | test("'"$CE5FE26_PATTERN"'"))]
}' "$INPUT_JSON" > "$CE5FE26_OUTPUT"

CE5FE26_COUNT=$(jq '.summary.total_structures' "$CE5FE26_OUTPUT")
CE5FE26_PASSED=$(jq '.summary.passed_prescreening' "$CE5FE26_OUTPUT")
CE5FE26_FAILED=$(jq '.summary.failed_prescreening' "$CE5FE26_OUTPUT")

echo "  Total: $CE5FE26_COUNT structures"
echo "  Passed: $CE5FE26_PASSED"
echo "  Failed: $CE5FE26_FAILED"
echo ""

echo "Processing Ce2Fe14B derivatives..."
echo "  Pattern: $CE2FE14B_PATTERN"
echo "  Output: $CE2FE14B_OUTPUT"

# Extract Ce2Fe14B derivatives with updated summary
jq '{
  summary: {
    total_structures: ([.results[] | select(.structure_id | test("'"$CE2FE14B_PATTERN"'"))] | length),
    passed_prescreening: ([.results[] | select(.structure_id | test("'"$CE2FE14B_PATTERN"'") and .passed_prescreening == true)] | length),
    failed_prescreening: ([.results[] | select(.structure_id | test("'"$CE2FE14B_PATTERN"'") and .passed_prescreening == false)] | length),
    hull_threshold: .summary.hull_threshold,
    energy_reference: .summary.energy_reference,
    prototype: "Ce2Fe14B",
    filtered_from: "'"$INPUT_JSON"'"
  },
  results: [.results[] | select(.structure_id | test("'"$CE2FE14B_PATTERN"'"))]
}' "$INPUT_JSON" > "$CE2FE14B_OUTPUT"

CE2FE14B_COUNT=$(jq '.summary.total_structures' "$CE2FE14B_OUTPUT")
CE2FE14B_PASSED=$(jq '.summary.passed_prescreening' "$CE2FE14B_OUTPUT")
CE2FE14B_FAILED=$(jq '.summary.failed_prescreening' "$CE2FE14B_OUTPUT")

echo "  Total: $CE2FE14B_COUNT structures"
echo "  Passed: $CE2FE14B_PASSED"
echo "  Failed: $CE2FE14B_FAILED"
echo ""

# Verification
echo "========================================================================"
echo "Verification"
echo "========================================================================"

ORIGINAL_TOTAL=$(jq '.summary.total_structures' "$INPUT_JSON")
SPLIT_TOTAL=$((CE5FE26_COUNT + CE2FE14B_COUNT))

echo "Original file total: $ORIGINAL_TOTAL structures"
echo "Split files total: $SPLIT_TOTAL structures"
echo "  Ce5Fe26: $CE5FE26_COUNT"
echo "  Ce2Fe14B: $CE2FE14B_COUNT"

if [ "$SPLIT_TOTAL" -eq "$ORIGINAL_TOTAL" ]; then
    echo ""
    echo "  SUCCESS: All structures accounted for"
elif [ "$SPLIT_TOTAL" -lt "$ORIGINAL_TOTAL" ]; then
    UNMATCHED=$((ORIGINAL_TOTAL - SPLIT_TOTAL))
    echo ""
    echo "  WARNING: $UNMATCHED structures did not match either pattern"
    echo ""
    echo "Unmatched structure IDs:"
    jq -r '.results[] | select(
        (.structure_id | test("'"$CE5FE26_PATTERN"'") | not) and
        (.structure_id | test("'"$CE2FE14B_PATTERN"'") | not)
    ) | .structure_id' "$INPUT_JSON" | head -20
else
    echo ""
    echo "  ERROR: Split total exceeds original (overlapping patterns?)"
fi

echo ""
echo "========================================================================"
echo "Sample Structures from Each File"
echo "========================================================================"

echo ""
echo "Ce5Fe26 derivatives (first 5):"
jq -r '.results[0:5] | .[] | "  \(.structure_id) - E_hull: \(.energy_above_hull // "N/A") eV/atom - Passed: \(.passed_prescreening)"' "$CE5FE26_OUTPUT"

echo ""
echo "Ce2Fe14B derivatives (first 5):"
jq -r '.results[0:5] | .[] | "  \(.structure_id) - E_hull: \(.energy_above_hull // "N/A") eV/atom - Passed: \(.passed_prescreening)"' "$CE2FE14B_OUTPUT"

echo ""
echo "========================================================================"
echo "Output Files Created"
echo "========================================================================"
echo "  $CE5FE26_OUTPUT"
echo "  $CE2FE14B_OUTPUT"
echo ""
echo "Use these files with:"
echo "  - Further analysis scripts"
echo "  - Workflow submission (only passed structures)"
echo "  - Statistical comparison between prototypes"
echo "========================================================================"

