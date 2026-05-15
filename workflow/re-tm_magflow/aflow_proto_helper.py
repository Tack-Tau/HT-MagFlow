#!/usr/bin/env python3
"""
AFLOW prototype matching helper for magnet candidates (R-T or R-T-T').

For each structure:
  1. Get ANRL prototype label via `aflow --prototype`
  2. Try direct match via `aflow --compare2prototypes`
  3. If no direct match and ternary: merge T+T' into one species,
     try binary match via `aflow --compare2prototypes --ignore_symmetry`
  4. Report: aflow_proto (ANRL stoichiometry) and pearson_symbol

Convention: A = rare earth, B = transition metal (1), C = transition metal (2)
"""

import subprocess
import tempfile
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from math import gcd
from concurrent.futures import ProcessPoolExecutor, as_completed

RARE_EARTH = {
    'Y', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sc'
}


def run_aflow_prototype(vasp_path):
    """Run `aflow --prototype` and return (label, pearson, stoich)."""
    try:
        result = subprocess.run(
            ['aflow', '--prototype'],
            stdin=open(vasp_path),
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout
    except Exception:
        return None, None, None

    label = None
    for line in output.splitlines():
        if line.startswith('AFLOW label'):
            label = line.split(':', 1)[1].strip()
            break

    if not label:
        return None, None, None

    parts = label.split('_')
    stoich = parts[0] if len(parts) >= 1 else ''
    pearson = parts[1] if len(parts) >= 2 else ''
    return label, pearson, stoich


def run_compare2prototypes(vasp_path, ignore_symmetry=False):
    """Run `aflow --compare2prototypes` and return True if a match is found."""
    cmd = ['aflow', '--compare2prototypes', '--catalog=all', '--screen_only']
    if ignore_symmetry:
        cmd.append('--ignore_symmetry')
    try:
        result = subprocess.run(
            cmd,
            stdin=open(vasp_path),
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
    except Exception:
        return False

    if 'No compatible prototypes found' in output:
        return False

    # Check if any duplicate found with misfit < 0.1
    for line in output.splitlines():
        line = line.strip()
        # Lines like "  212                    0.0784532"
        match = re.search(r'\s+([\d.]+)\s*$', line)
        if match:
            try:
                misfit = float(match.group(1))
                if misfit < 0.1:
                    return True
            except ValueError:
                continue
    return False


def make_binary_poscar(vasp_path):
    """Create a binary POSCAR by merging all TM species into one.
    Returns path to temp file, or None on failure."""
    with open(vasp_path) as f:
        lines = f.readlines()

    # Find element line and count line (lines 5 and 6 in standard VASP5 format)
    elem_line = lines[5].split()
    count_line = [int(x) for x in lines[6].split()]

    if len(elem_line) != len(count_line):
        return None

    re_idx = [i for i, e in enumerate(elem_line) if e in RARE_EARTH]
    tm_idx = [i for i, e in enumerate(elem_line) if e not in RARE_EARTH]

    if not re_idx or not tm_idx:
        return None

    re_count = sum(count_line[i] for i in re_idx)
    tm_count = sum(count_line[i] for i in tm_idx)

    # Parse coordinate block (line 7 = Direct/Cartesian, line 8+ = coords)
    coord_start = 8
    coords_by_species = []
    idx = coord_start
    for i, cnt in enumerate(count_line):
        species_coords = []
        for _ in range(cnt):
            if idx < len(lines) and lines[idx].strip() and not lines[idx].strip().startswith('0.00000000E'):
                species_coords.append(lines[idx])
            elif idx < len(lines) and lines[idx].strip():
                break
            idx += 1
        coords_by_species.append(species_coords)

    # Reorder: RE atoms first, then all TM atoms merged
    re_coords = []
    for i in re_idx:
        re_coords.extend(coords_by_species[i])
    tm_coords = []
    for i in tm_idx:
        tm_coords.extend(coords_by_species[i])

    if len(re_coords) != re_count or len(tm_coords) != tm_count:
        # Fallback: just read all coords sequentially
        all_coords = []
        idx = coord_start
        total = sum(count_line)
        for _ in range(total):
            if idx < len(lines) and lines[idx].strip():
                all_coords.append(lines[idx])
                idx += 1
        # Split by cumulative counts
        re_coords = []
        tm_coords = []
        cum = 0
        for i, cnt in enumerate(count_line):
            for j in range(cnt):
                if i in re_idx:
                    re_coords.append(all_coords[cum + j])
                else:
                    tm_coords.append(all_coords[cum + j])
            cum += cnt

    # Build new binary POSCAR
    re_elem = elem_line[re_idx[0]]
    new_lines = lines[0:5]
    new_lines.append(f'   {re_elem}   X\n')
    new_lines.append(f'   {re_count}   {tm_count}\n')
    new_lines.append(lines[7])  # Direct/Cartesian
    new_lines.extend(re_coords)
    new_lines.extend(tm_coords)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.vasp', delete=False)
    tmp.write(''.join(new_lines))
    tmp.close()
    return tmp.name


def get_binary_stoich(vasp_path):
    """Get the reduced binary stoichiometry A_xB_y from the POSCAR."""
    with open(vasp_path) as f:
        lines = f.readlines()
    elem_line = lines[5].split()
    count_line = [int(x) for x in lines[6].split()]

    re_idx = [i for i, e in enumerate(elem_line) if e in RARE_EARTH]
    tm_idx = [i for i, e in enumerate(elem_line) if e not in RARE_EARTH]

    re_count = sum(count_line[i] for i in re_idx)
    tm_count = sum(count_line[i] for i in tm_idx)

    g = gcd(re_count, tm_count)
    a = re_count // g
    b = tm_count // g
    return a, b


def format_anrl(counts, labels='ABC'):
    """Format counts dict into ANRL stoichiometry string (e.g. {1:2, 2:17} -> 'A2B17')."""
    parts = []
    for i, cnt in enumerate(counts):
        lbl = labels[i]
        if cnt == 1:
            parts.append(lbl)
        else:
            parts.append(f'{lbl}{cnt}')
    return ''.join(parts)


def get_our_stoich(vasp_path):
    """
    Read POSCAR and return canonical stoichiometry with our convention:
    A = rare earth, B = TM (or TM1), C = TM2.

    For binary R-T: returns e.g. 'AB5'
    For ternary R-T-T': returns e.g. 'A2B16C'
    """
    with open(vasp_path) as f:
        lines = f.readlines()
    elem_line = lines[5].split()
    count_line = [int(x) for x in lines[6].split()]

    re_elems = [(e, c) for e, c in zip(elem_line, count_line) if e in RARE_EARTH]
    tm_elems = [(e, c) for e, c in zip(elem_line, count_line) if e not in RARE_EARTH]

    re_count = sum(c for _, c in re_elems)
    tm_counts = [c for _, c in tm_elems]

    all_counts = [re_count] + tm_counts
    g = all_counts[0]
    for c in all_counts[1:]:
        g = gcd(g, c)
    reduced = [c // g for c in all_counts]

    return format_anrl(reduced)


def process_structure(vasp_path):
    """Process one structure and return (aflow_proto, pearson_symbol, match_type)."""
    sid = Path(vasp_path).stem

    # Step 1: Get ANRL label and Pearson from aflow --prototype
    aflow_label, pearson, _ = run_aflow_prototype(vasp_path)
    if not aflow_label:
        return sid, '', '', 'failed'

    # Compute stoichiometry with our convention (A=RE, B=TM1, C=TM2)
    our_stoich = get_our_stoich(vasp_path)

    # Determine number of species
    with open(vasp_path) as f:
        lines = f.readlines()
    n_species = len(lines[5].split())

    # Step 2: Try direct match against AFLOW prototype encyclopedia
    direct_match = run_compare2prototypes(vasp_path, ignore_symmetry=False)
    if direct_match:
        match_type = 'ternary' if n_species >= 3 else 'binary'
        return sid, our_stoich, pearson, match_type

    # Step 3: For ternary (3+ species): merge TMs, try binary match
    if n_species >= 3:
        binary_path = make_binary_poscar(vasp_path)
        if binary_path:
            try:
                binary_match = run_compare2prototypes(binary_path, ignore_symmetry=True)
                if binary_match:
                    a, b = get_binary_stoich(vasp_path)
                    _, binary_pearson, _ = run_aflow_prototype(binary_path)
                    final_pearson = binary_pearson if binary_pearson else pearson
                    proto = format_anrl([a, b])
                    return sid, proto, final_pearson, 'binary_merge'
            finally:
                os.unlink(binary_path)

    # Step 4: No match
    return sid, our_stoich, pearson, 'new'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--struct-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--merged', type=Path, required=True)
    parser.add_argument('--np', type=int, default=1)
    args = parser.parse_args()

    vasp_files = sorted(args.struct_dir.glob('*.vasp'))
    n_struct = len(vasp_files)
    print(f"  N structures:  {n_struct}")

    # Read input CSV
    with open(args.input) as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)
    input_by_id = {row['structure_id']: row for row in input_rows}

    # Verify structure_id consistency
    vasp_ids = {p.stem for p in vasp_files}
    csv_ids = set(input_by_id.keys())
    missing_vasp = csv_ids - vasp_ids
    if missing_vasp:
        print(f"  WARNING: {len(missing_vasp)} IDs in CSV but no .vasp file")
        for sid in sorted(missing_vasp)[:5]:
            print(f"    {sid}")

    # Process all structures
    results = {}
    count = 0

    if args.np <= 1:
        for vasp_file in vasp_files:
            count += 1
            if count % 10 == 0 or count == n_struct:
                print(f"  Processing {count} / {n_struct} ...", flush=True)
            sid, proto, pearson, match_type = process_structure(str(vasp_file))
            results[sid] = (proto, pearson, match_type)
    else:
        with ProcessPoolExecutor(max_workers=args.np) as executor:
            futures = {executor.submit(process_structure, str(f)): f for f in vasp_files}
            for future in as_completed(futures):
                count += 1
                if count % 10 == 0 or count == n_struct:
                    print(f"  Processing {count} / {n_struct} ...", flush=True)
                sid, proto, pearson, match_type = future.result()
                results[sid] = (proto, pearson, match_type)

    # Write full AFLOW output CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['structure_id', 'aflow_proto', 'pearson_symbol', 'match_type'])
        for sid in sorted(results.keys()):
            proto, pearson, match_type = results[sid]
            writer.writerow([sid, proto, pearson, match_type])

    # Write merged CSV (input + aflow_proto + pearson_symbol + match_type columns)
    with open(args.merged, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'structure_id', 'mattersim_e_hull', 'dft_e_hull',
            'mattersim_energy_per_atom', 'vasp_energy_per_atom',
            'spg_num', 'aflow_proto', 'pearson_symbol', 'match_type'
        ])
        for row in input_rows:
            sid = row['structure_id']
            if sid in results:
                proto, pearson, match_type = results[sid]
            else:
                proto, pearson, match_type = '', '', ''
            writer.writerow([
                sid, row['mattersim_e_hull'], row['dft_e_hull'],
                row['mattersim_energy_per_atom'], row['vasp_energy_per_atom'],
                row['spg_num'], proto, pearson, match_type
            ])

    # Summary
    n_ternary = sum(1 for v in results.values() if v[2] == 'ternary')
    n_binary = sum(1 for v in results.values() if v[2] == 'binary')
    n_binary_merge = sum(1 for v in results.values() if v[2] == 'binary_merge')
    n_new = sum(1 for v in results.values() if v[2] == 'new')
    n_failed = sum(1 for v in results.values() if v[2] == 'failed')

    print(f"\n{'='*70}")
    print(f"--- Results Summary ---")
    print(f"  Total:          {n_struct}")
    print(f"  Ternary match:  {n_ternary}")
    print(f"  Binary match:   {n_binary}")
    print(f"  Binary merge:   {n_binary_merge}")
    print(f"  New prototype:  {n_new}")
    print(f"  Failed:         {n_failed}")
    print(f"\n  Output:  {args.output}")
    print(f"  Merged:  {args.merged}")
    print(f"{'='*70}")

    # Top prototypes
    from collections import Counter
    proto_counts = Counter(v[0] for v in results.values() if v[0])
    print(f"\nTop 20 most common aflow_proto:")
    for proto, cnt in proto_counts.most_common(20):
        print(f"  {cnt:4d}  {proto}")

    pearson_counts = Counter(v[1] for v in results.values() if v[1])
    print(f"\nTop 20 most common Pearson symbols:")
    for p, cnt in pearson_counts.most_common(20):
        print(f"  {cnt:4d}  {p}")


if __name__ == '__main__':
    main()
