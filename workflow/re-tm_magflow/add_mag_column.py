#!/usr/bin/env python3
"""
Add total magnetization (mu_B/cell) and cell volume (A^3) columns to candidates.csv
by reading the final relaxation step from VASP OUTCAR files.

Total magnetization is read from the "number of electron ... magnetization" line
in OUTCAR (VASP's integrated charge density magnetization).
Cell volume is read from the last "volume of cell" line in OUTCAR.

Usage:
    python3 add_mag_column.py --csv ter_mag_results/candidates.csv \
                              --vasp-dir ./VASP_JOBS \
                              --output ter_mag_results/candidates_mag.csv
"""

import argparse
import csv
import os
import re
import sys


MAG_PATTERN = re.compile(r'number of electron\s+\S+\s+magnetization\s+(\S+)')


def parse_outcar(outcar_path):
    """
    Extract total magnetization and cell volume from OUTCAR.

    Reads the last occurrence of each pattern (corresponding to the final ionic step).

    Returns:
        (total_mag, volume) or (None, None) if parsing fails.
    """
    if not os.path.isfile(outcar_path):
        return None, None

    total_mag = None
    volume = None

    with open(outcar_path, 'r') as f:
        for line in f:
            if 'volume of cell' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        volume = float(parts[1].strip())
                    except ValueError:
                        pass
            elif 'magnetization' in line:
                m = MAG_PATTERN.search(line)
                if m:
                    total_mag = float(m.group(1))

    return total_mag, volume


def main():
    parser = argparse.ArgumentParser(
        description="Add magnetization and volume columns to candidates CSV"
    )
    parser.add_argument(
        '--csv', required=True,
        help='Path to input candidates.csv'
    )
    parser.add_argument(
        '--vasp-dir', default='./VASP_JOBS',
        help='Base VASP jobs directory (default: ./VASP_JOBS)'
    )
    parser.add_argument(
        '--output', default=None,
        help='Output CSV path (default: overwrite input)'
    )
    args = parser.parse_args()

    output_path = args.output or args.csv

    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    new_fields = fieldnames + ['total_mag', 'cell_volume', 'mag_per_vol']

    success = 0
    missing = 0

    for row in rows:
        struct_id = row['structure_id']
        comp = struct_id.rsplit('_s', 1)[0]
        outcar_path = os.path.join(args.vasp_dir, comp, struct_id, 'Relax', 'OUTCAR')

        total_mag, volume = parse_outcar(outcar_path)

        if total_mag is not None:
            row['total_mag'] = f"{total_mag:.3f}"
            row['cell_volume'] = f"{volume:.2f}"
            row['mag_per_vol'] = f"{total_mag / volume:.4f}"
            success += 1
        else:
            row['total_mag'] = ''
            row['cell_volume'] = ''
            row['mag_per_vol'] = ''
            missing += 1
            print(f"  WARNING: Could not parse OUTCAR for {struct_id}: {outcar_path}")

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: {success} parsed, {missing} missing/failed")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
