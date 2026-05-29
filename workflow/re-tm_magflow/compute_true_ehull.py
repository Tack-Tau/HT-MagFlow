#!/usr/bin/env python3
"""
Compute true E_hull-DFT for all candidates by building the full convex hull
(MP reference phases + all VASP-relaxed generated structures) and insert
a 'true_dft_e_hull' column right after 'dft_e_hull' in each
candidates_w_proto_mag.csv.

Reuses the same data loading logic as plot_bin_mag_hull.py / plot_ter_mag_hull.py.
"""

import json
import csv
import sys
from pathlib import Path

from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry


def load_mp_ref_entries(path):
    with open(path) as f:
        data = json.load(f)
    entries = []
    for item in data:
        comp = Composition(item['composition'])
        entries.append(PDEntry(comp, item['energy'],
                               name=item.get('mp_id', item.get('entry_id', ''))))
    return entries


def detect_ref_source(dft_json):
    with open(dft_json) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'summary' in data:
        src = data['summary'].get('reference_phase_source', '')
        if src == 'MatterSim':
            return 'mattersim'
        if src == 'Materials Project':
            return 'mp'
    raise ValueError(f"Unknown reference_phase_source in {dft_json}")


def load_dft_gen_entries(path):
    with open(path) as f:
        data = json.load(f)
    results = data.get('results', data) if isinstance(data, dict) else data
    entries = []
    for r in results:
        epa = r.get('vasp_energy_per_atom')
        if epa is None:
            continue
        comp = Composition(r['composition'])
        total_e = epa * comp.num_atoms
        entries.append(PDEntry(comp, total_e, name=r['structure_id']))
    return entries


def filter_for_chemsys(entries, chemsys_elements):
    return [e for e in entries
            if set(str(el) for el in e.composition.elements).issubset(chemsys_elements)]


def process_results_dir(rdir):
    rdir = Path(rdir)
    csv_path = rdir / 'candidates_w_proto_mag.csv'
    dft_json = rdir / 'dft_stability_results.json'

    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        return
    if not dft_json.exists():
        print(f"  SKIP: {dft_json} not found")
        return

    ref = detect_ref_source(dft_json)
    mp_json = rdir / ('mp_mattersim.json' if ref == 'mattersim' else 'mp_vaspdft.json')
    if not mp_json.exists():
        print(f"  SKIP: {mp_json} not found")
        return

    print(f"  MP ref: {mp_json.name} ({ref})")

    mp_entries = load_mp_ref_entries(mp_json)
    gen_entries = load_dft_gen_entries(dft_json)
    all_entries = mp_entries + gen_entries

    print(f"  MP entries: {len(mp_entries)}, generated entries: {len(gen_entries)}")

    # Build per-structure_id lookup: structure_id -> PDEntry
    sid_to_entry = {}
    for e in gen_entries:
        sid_to_entry[e.name] = e

    # Detect all chemical systems present in generated entries
    chemsys_set = set()
    for e in gen_entries:
        elems = frozenset(str(el) for el in e.composition.elements)
        chemsys_set.add(elems)

    # Build phase diagram per chemical system and compute true e_hull
    sid_to_true_ehull = {}
    for elems in chemsys_set:
        sub_entries = filter_for_chemsys(all_entries, elems)
        if len(sub_entries) < 2:
            continue
        # Need elemental references
        elem_symbols = set()
        for e in sub_entries:
            if len(e.composition.elements) == 1:
                elem_symbols.add(str(e.composition.elements[0]))
        if not elems.issubset(elem_symbols):
            continue
        try:
            pd = PhaseDiagram(sub_entries)
        except Exception:
            continue
        for e in sub_entries:
            if e.name in sid_to_entry:
                sid_to_true_ehull[e.name] = pd.get_e_above_hull(e)

    # Read CSV, insert column, write back
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Find dft_e_hull column index
    dft_col = header.index('dft_e_hull')
    insert_at = dft_col + 1

    if 'true_dft_e_hull' in header:
        true_col = header.index('true_dft_e_hull')
        for row in rows:
            sid = row[0]
            row[true_col] = str(sid_to_true_ehull.get(sid, ''))
    else:
        header.insert(insert_at, 'true_dft_e_hull')
        for row in rows:
            sid = row[0]
            val = sid_to_true_ehull.get(sid, '')
            row.insert(insert_at, str(val))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    n_filled = sum(1 for sid in sid_to_true_ehull if sid in sid_to_entry)
    n_on_hull = sum(1 for v in sid_to_true_ehull.values() if abs(v) < 1e-10)
    print(f"  Wrote true_dft_e_hull for {n_filled} candidates ({n_on_hull} on true hull)")


def main():
    base = Path(__file__).parent
    dirs = [
        base / 'bin_mag_results',
        base / 'ter_mag_results',
        base / 'new_bin_mag_results',
        base / 'new_ter_mag_results',
    ]
    for d in dirs:
        print(f"\n=== {d.name} ===")
        process_results_dir(d)
    print("\nDone.")


if __name__ == '__main__':
    main()
