#!/bin/bash
# Extract Wyckoff positions for binary_merge ternary candidates using aflow --wyccar.
# Produces a TSV: structure_id, spg_num, aflow_anrl, element, multiplicity, wyckoff_letter, site_symmetry
# Usage: bash analyze_wyckoff.sh [output_file]

OUTPUT="${1:-wyckoff_analysis.tsv}"
BASE="$(cd "$(dirname "$0")" && pwd)"

RESULT_DIRS=(
    "$BASE/ter_mag_results"
    "$BASE/new_ter_mag_results"
)

echo -e "structure_id\tspg_num\taflow_anrl\telement\tmultiplicity\twyckoff_letter\tsite_symmetry" > "$OUTPUT"

for DIR in "${RESULT_DIRS[@]}"; do
    CSV="$DIR/candidates_w_proto_mag.csv"
    STRUCT_DIR="$DIR/struct"
    [ -f "$CSV" ] || continue
    [ -d "$STRUCT_DIR" ] || continue

    # Extract binary_merge rows: structure_id (col1), spg_num (col6), aflow_anrl (col8)
    tail -n +2 "$CSV" | awk -F',' '$10=="binary_merge" {print $1","$6","$8}' | while IFS=',' read -r SID SPG ANRL; do
        VASP_FILE="$STRUCT_DIR/${SID}.vasp"
        [ -f "$VASP_FILE" ] || continue

        # Run aflow --wyccar and parse the Direct(WYCCAR) lines
        aflow --wyccar < "$VASP_FILE" 2>/dev/null | awk -v sid="$SID" -v spg="$SPG" -v anrl="$ANRL" '
            /^Direct\(WYCCAR\)/ { reading=1; next }
            reading && NF>=6 {
                elem=$4; mult=$5; wyck=$6; sitesym=$7
                print sid "\t" spg "\t" anrl "\t" elem "\t" mult "\t" wyck "\t" sitesym
            }
        ' >> "$OUTPUT"
    done
done

N=$(tail -n +2 "$OUTPUT" | wc -l)
echo "Wrote $N Wyckoff entries to $OUTPUT"
