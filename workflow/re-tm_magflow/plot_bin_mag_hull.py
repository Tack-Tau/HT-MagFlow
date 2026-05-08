#!/usr/bin/env python3
"""
Plot binary convex hull diagrams for binary magnet results.

For each binary chemsys, builds two independent convex hulls:
  1. MP-ref hull: reference phases only (from mp_vaspdft.json or mp_mattersim.json)
  2. DFT hull:    reference phases + VASP-relaxed generated structures

The MP phase energy reference is auto-detected from dft_stability_results.json
(field reference_phase_source): uses mp_mattersim.json when MatterSim,
mp_vaspdft.json otherwise.

Usage:
    python3 plot_bin_mag_hull.py --results-dir bin_mag_results
    python3 plot_bin_mag_hull.py --results-dir bin_mag_results --db bin_mag_results/prescreening_structures.db
    python3 plot_bin_mag_hull.py --results-dir bin_mag_results --e-above-hull-max 0.05 --output-dir hull_plots
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='ase')

import json
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry


def formula_to_latex(formula):
    def replace_elem(m):
        el, cnt = m.group(1), m.group(2)
        return el if cnt == '1' else f'{el}$_{{{cnt}}}$'
    return re.sub(r'([A-Z][a-z]?)(\d+)', replace_elem, formula)


def is_metal(symbol):
    try:
        e = Element(symbol)
        return (e.is_metal or e.is_alkali or e.is_alkaline
                or e.is_transition_metal or e.is_post_transition_metal
                or e.is_rare_earth_metal or e.is_actinoid)
    except Exception:
        return False


def load_pearson_from_db(db_path, structure_ids):
    """Load Pearson symbols for given structure IDs from a PyXtal/ASE database."""
    from ase.db import connect
    db = connect(str(db_path))
    pearson = {}
    for sid in structure_ids:
        try:
            row = db.get(structure_id=sid)
            pearson[sid] = row.get('pearson_symbol', '')
        except KeyError:
            pass
    return pearson


def load_mp_ref_entries(path):
    """Load MP reference phase entries (total energies) from a JSON list."""
    with open(path) as f:
        data = json.load(f)
    entries = []
    for item in data:
        comp = Composition(item['composition'])
        entries.append(PDEntry(comp, item['energy'],
                               name=item.get('mp_id', item.get('entry_id', ''))))
    return entries


def detect_ref_source(dft_json):
    """Read reference_phase_source from dft_stability_results.json summary.

    compute_dft_e_hull.py writes exactly 'MatterSim' or 'Materials Project'.
    """
    with open(dft_json) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'summary' in data:
        src = data['summary'].get('reference_phase_source', '')
        if src == 'MatterSim':
            return 'mattersim'
        if src == 'Materials Project':
            return 'mp'
    raise ValueError(f"Unknown reference_phase_source in {dft_json}: '{src}'")


def load_dft_gen_entries(path):
    """Load VASP-relaxed generated structures from dft_stability_results.json."""
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


def filter_for_chemsys(entries, chemsys):
    elements = set(chemsys.split('-'))
    return [e for e in entries
            if set(str(el) for el in e.composition.elements).issubset(elements)]


def detect_binary_systems(entries):
    systems = set()
    for e in entries:
        elems = sorted(str(el) for el in e.composition.elements)
        if len(elems) == 2:
            systems.add('-'.join(elems))
    return sorted(systems)


def plot_binary_hull(chemsys, mp_entries_sys, gen_entries_sys, ax, e_hull_max,
                     near_threshold=0.01, pearson_map=None):
    """Plot one binary hull on the given axis."""
    elements = sorted(chemsys.split('-'))
    if is_metal(elements[1]) and not is_metal(elements[0]):
        elements = [elements[1], elements[0]]

    if len(mp_entries_sys) < 2:
        ax.text(0.5, 0.5, f'{chemsys}\nInsufficient MP data',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return

    elem_set = set(elements)
    mp_elems = set()
    for e in mp_entries_sys:
        if len(e.composition.elements) == 1:
            mp_elems.add(str(e.composition.elements[0]))
    if not elem_set.issubset(mp_elems):
        ax.text(0.5, 0.5, f'{chemsys}\nMissing elemental refs',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return

    try:
        pd_mp = PhaseDiagram(mp_entries_sys)
    except Exception as exc:
        ax.text(0.5, 0.5, f'{chemsys}\nPD error: {str(exc)[:40]}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return

    all_entries = mp_entries_sys + gen_entries_sys
    try:
        pd_all = PhaseDiagram(all_entries)
    except Exception:
        pd_all = pd_mp

    el_right = elements[1]

    def x_of(entry):
        return entry.composition.get_atomic_fraction(el_right)

    # Classify MP phases: on combined hull and/or on MP-only hull
    mp_stable = []
    for e in mp_entries_sys:
        x = x_of(e)
        fe = pd_all.get_form_energy_per_atom(e)
        on_hull = e in pd_all.stable_entries
        on_mp_hull = e in pd_mp.stable_entries
        mp_stable.append({'x': x, 'fe': fe, 'entry': e,
                          'on_hull': on_hull, 'on_mp_hull': on_mp_hull,
                          'formula': e.composition.reduced_formula})

    # Classify generated structures
    stable_gen = []
    meta_gen = []
    near_meta_gen = []
    near_threshold = min(near_threshold, e_hull_max)
    for e in gen_entries_sys:
        if len(e.composition.elements) != 2:
            continue
        x = x_of(e)
        fe = pd_all.get_form_energy_per_atom(e)
        e_hull = pd_all.get_e_above_hull(e)
        rec = {'x': x, 'fe': fe, 'entry': e, 'e_hull': e_hull,
               'formula': e.composition.reduced_formula}
        if e in pd_all.stable_entries:
            stable_gen.append(rec)
        elif e_hull <= near_threshold:
            near_meta_gen.append(rec)
        elif e_hull <= e_hull_max:
            meta_gen.append(rec)

    if pearson_map is None:
        pearson_map = {}

    # -- Hull lines --
    all_hull_pts = sorted(
        [(x_of(e), pd_all.get_form_energy_per_atom(e)) for e in pd_all.stable_entries],
        key=lambda p: p[0])
    ax.plot([p[0] for p in all_hull_pts], [p[1] for p in all_hull_pts],
            color='black', ls='-', lw=3, zorder=2, label='Convex hull')

    if stable_gen:
        mp_hull_pts = sorted(
            [(x_of(e), pd_mp.get_form_energy_per_atom(e)) for e in pd_mp.stable_entries],
            key=lambda p: p[0])
        ax.plot([p[0] for p in mp_hull_pts], [p[1] for p in mp_hull_pts],
                color='gray', ls='--', lw=2.5, alpha=0.6, zorder=1,
                label='MP-only hull')

    # -- MP phases on either hull: each with distinct tab20 color, circle, legend --
    mp_on_any_hull = [d for d in mp_stable if d['on_hull'] or d['on_mp_hull']]
    mp_colors = plt.cm.tab20(np.linspace(0, 1, max(len(mp_on_any_hull), 1)))
    for i, d in enumerate(mp_on_any_hull):
        color = mp_colors[i % len(mp_colors)]
        ax.scatter(d['x'], d['fe'], c=[color], s=150, marker='o',
                   edgecolors='black', linewidth=1.5, zorder=3,
                   label=formula_to_latex(d['formula']))

    # -- New stable structures: each with distinct marker, blue/red, individual legend --
    STABLE_MARKERS = ['^', 'D', 'v', 'p', 'P', 'X', 'h', 'H', '<', '>']
    for i, d in enumerate(stable_gen):
        marker = STABLE_MARKERS[i % len(STABLE_MARKERS)]
        ax.scatter(d['x'], d['fe'], c='blue', s=150, marker=marker,
                   edgecolors='red', linewidth=2.5, zorder=5)
        sid = d['entry'].name
        pearson = pearson_map.get(sid, '')
        formula_latex = formula_to_latex(d['formula'])
        label = f'{pearson}-{formula_latex}' if pearson else formula_latex
        ax.scatter([], [], c='blue', s=150, marker=marker,
                   edgecolors='red', linewidth=2.5, label=label)

    # -- Near-metastable structures (within 0.2*e_hull_max): distinct colors, legend --
    near_colors = plt.cm.Set2(np.linspace(0, 1, max(len(near_meta_gen), 1)))
    for i, d in enumerate(near_meta_gen):
        color = near_colors[i % len(near_colors)]
        ax.scatter(d['x'], d['fe'], c=[color], s=120, marker='s',
                   edgecolors='red', linewidth=2, zorder=4)
        sid = d['entry'].name
        pearson = pearson_map.get(sid, '')
        formula_latex = formula_to_latex(d['formula'])
        label = f'{pearson}-{formula_latex}' if pearson else formula_latex
        ax.scatter([], [], c=[color], s=120, marker='s',
                   edgecolors='red', linewidth=2, label=label)

    # -- Remaining metastable structures: orange squares, no legend --
    for d in meta_gen:
        ax.scatter(d['x'], d['fe'], c='orange', s=120, marker='s',
                   edgecolors='red', linewidth=2, zorder=4)

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(f'x({elements[1]}) in {elements[0]}$_{{1-x}}${elements[1]}$_x$',
                  fontsize=20, fontweight='bold')
    ax.set_ylabel('Formation Energy (eV/atom)', fontsize=20, fontweight='bold')
    if meta_gen:
        title = f'{chemsys} ($E_{{\\mathrm{{hull}}}} \\leq {e_hull_max:.2f}$ eV/atom)'
    else:
        title = chemsys
    ax.set_title(title, fontsize=24, fontweight='bold', pad=10)
    ax.legend(fontsize=18, loc='lower right', framealpha=0.65,
              edgecolor='black', facecolor='white')
    ax.grid(True, alpha=0.3, ls='--')
    ax.tick_params(axis='both', which='major', labelsize=18)


def main():
    parser = argparse.ArgumentParser(
        description='Plot binary convex hulls for binary magnet results')
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Directory with JSON result files')
    parser.add_argument('--db', type=Path, default=None,
                        help='Path to prescreening_structures.db (PyXtal/ASE) '
                             'for Pearson symbol lookup (optional)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: same as results-dir)')
    parser.add_argument('--e-above-hull-max', type=float, default=0.05,
                        help='Maximum energy above hull to display metastable '
                             'structures (eV/atom, default: 0.05)')
    parser.add_argument('--near-threshold', type=float, default=0.01,
                        help='Energy above hull threshold for near-metastable '
                             'structures shown with individual legend entries '
                             '(eV/atom, default: 0.01)')
    parser.add_argument('--systems', nargs='+', default=None,
                        help='Specific chemsys to plot (default: auto-detect all binary)')
    args = parser.parse_args()

    rdir = args.results_dir
    odir = args.output_dir or rdir
    odir.mkdir(parents=True, exist_ok=True)

    dft_json = rdir / 'dft_stability_results.json'
    if not dft_json.exists():
        print(f"ERROR: Required file not found: {dft_json}")
        return 1

    ref = detect_ref_source(dft_json)
    if ref == 'mattersim':
        mp_json = rdir / 'mp_mattersim.json'
        ref_label = 'MatterSim-DFT'
    else:
        mp_json = rdir / 'mp_vaspdft.json'
        ref_label = 'MP-GGA'
    print(f"MP energy reference: {ref_label} ({mp_json.name})")

    if not mp_json.exists():
        print(f"ERROR: Required file not found: {mp_json}")
        return 1

    print("Loading data...")
    mp_entries = load_mp_ref_entries(mp_json)
    gen_entries = load_dft_gen_entries(dft_json)
    print(f"  MP-ref source:   {mp_json.name} ({ref_label})")
    print(f"  MP-ref entries:  {len(mp_entries)}")
    print(f"  Generated (DFT): {len(gen_entries)}")
    print(f"  E-above-hull-max: {args.e_above_hull_max} eV/atom")
    print(f"  Near-threshold:   {args.near_threshold} eV/atom")

    pearson_map = {}
    if args.db:
        if args.db.exists():
            gen_ids = [e.name for e in gen_entries]
            pearson_map = load_pearson_from_db(args.db, gen_ids)
            print(f"  Pearson symbols:  {len(pearson_map)} loaded from {args.db.name}")
        else:
            print(f"  WARNING: DB not found: {args.db}, skipping Pearson lookup")

    if args.systems:
        systems = ['-'.join(sorted(s.split('-'))) for s in args.systems]
    else:
        gen_systems = detect_binary_systems(gen_entries)
        mp_systems = detect_binary_systems(mp_entries)
        systems = sorted(set(gen_systems) & set(mp_systems))

    print(f"  Binary systems:  {systems}\n")

    if not systems:
        print("No binary systems found.")
        return 0

    for chemsys in systems:
        mp_sub = filter_for_chemsys(mp_entries, chemsys)
        gen_sub = filter_for_chemsys(gen_entries, chemsys)
        print(f"  {chemsys}: {len(mp_sub)} MP-ref, {len(gen_sub)} generated")

        fig, ax = plt.subplots(figsize=(14, 8))
        plot_binary_hull(chemsys, mp_sub, gen_sub, ax, args.e_above_hull_max,
                         near_threshold=args.near_threshold,
                         pearson_map=pearson_map)
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.92)

        stem = chemsys.replace('-', '')
        png_path = odir / f'{stem}_hull.png'
        pdf_path = odir / f'{stem}_hull.pdf'
        fig.savefig(str(png_path), dpi=300)
        fig.savefig(str(pdf_path))
        plt.close(fig)

        print(f"    -> {png_path}")
        print(f"    -> {pdf_path}")

    print(f"\nAll {len(systems)} hull plots saved to: {odir.resolve()}")
    return 0


if __name__ == '__main__':
    exit(main())
