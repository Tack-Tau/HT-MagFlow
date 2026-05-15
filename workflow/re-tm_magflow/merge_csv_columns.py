#!/usr/bin/env python3
"""
Merge additional columns from a source CSV into a base CSV by structure_id.

Usage:
    python3 merge_csv_columns.py --base candidates_w_proto.csv \
                                 --source ter_mag_results/candidates_w_proto_mag.csv \
                                 --columns total_mag cell_volume mag_per_vol \
                                 --output candidates_w_proto_merged.csv
"""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description="Merge columns from source CSV into base CSV by structure_id")
    parser.add_argument('--base', required=True, help='Base CSV file')
    parser.add_argument('--source', required=True, help='Source CSV with additional columns')
    parser.add_argument('--columns', nargs='+', required=True, help='Column names to merge from source')
    parser.add_argument('--output', required=True, help='Output CSV file')
    args = parser.parse_args()

    with open(args.source) as f:
        reader = csv.DictReader(f)
        source_by_id = {}
        for row in reader:
            source_by_id[row['structure_id']] = row

    with open(args.base) as f:
        reader = csv.DictReader(f)
        base_fields = reader.fieldnames
        base_rows = list(reader)

    out_fields = base_fields + [c for c in args.columns if c not in base_fields]

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in base_rows:
            sid = row['structure_id']
            src = source_by_id.get(sid, {})
            for col in args.columns:
                row[col] = src.get(col, '')
            writer.writerow(row)

    print(f"Merged {len(args.columns)} columns into {args.output} ({len(base_rows)} rows)")


if __name__ == '__main__':
    main()
