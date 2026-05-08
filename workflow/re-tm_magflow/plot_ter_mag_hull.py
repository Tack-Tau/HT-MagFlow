#!/usr/bin/env python3
"""
Plot ternary convex hull diagrams for ternary magnet results.

For each ternary chemsys, builds two independent convex hulls:
  1. MP-ref hull: reference phases only (from mp_vaspdft.json or mp_mattersim.json)
  2. DFT hull:    reference phases + VASP-relaxed generated structures

The MP phase energy reference is auto-detected from dft_stability_results.json
(field reference_phase_source): uses mp_mattersim.json when MatterSim,
mp_vaspdft.json otherwise.

Usage:
    python3 plot_ter_mag_hull.py --results-dir ter_mag_results
    python3 plot_ter_mag_hull.py --results-dir ter_mag_results --db ter_mag_results/prescreening_structures.db
    python3 plot_ter_mag_hull.py --results-dir ter_mag_results --e-above-hull-max 0.05 --systems Co-Cr-Gd Fe-Ti-Y
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
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from scipy.spatial import Delaunay
from pathlib import Path

from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry


def formula_to_latex(formula):
    def replace_paren(m):
        cnt = m.group(1)
        return ')' if cnt == '1' else f')$_{{{cnt}}}$'
    result = re.sub(r'\)(\d+)', replace_paren, formula)
    def replace_elem(m):
        el, cnt = m.group(1), m.group(2)
        return el if cnt == '1' else f'{el}$_{{{cnt}}}$'
    return re.sub(r'([A-Z][a-z]?)(\d+)', replace_elem, result)


def bary2cart(bary):
    """Convert barycentric (a, b, c) to 2D Cartesian for equilateral triangle."""
    a, b, c = bary
    x = 0.5 * a + c
    y = np.sqrt(3) / 2 * a
    return np.array([x, y])


def draw_ternary_axes(ax, elements):
    sqrt3 = np.sqrt(3)
    vertices = np.array([[0.5, sqrt3 / 2], [0, 0], [1, 0]])
    tri = Polygon(vertices, fill=False, edgecolor='black', linewidth=2.5)
    ax.add_patch(tri)

    offset = 0.06
    ax.text(vertices[0, 0], vertices[0, 1] + offset, elements[0],
            ha='center', va='bottom', fontsize=22, fontweight='bold')
    ax.text(vertices[1, 0] - offset, vertices[1, 1], elements[1],
            ha='right', va='center', fontsize=22, fontweight='bold')
    ax.text(vertices[2, 0] + offset, vertices[2, 1], elements[2],
            ha='left', va='center', fontsize=22, fontweight='bold')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.12, sqrt3 / 2 + 0.12)


def draw_hull_facets(ax, pd_obj, entry_cart, label=None, zorder=1, **kwargs):
    """Draw Delaunay triangulation edges for stable entries of a PhaseDiagram."""
    coords = []
    for e in pd_obj.stable_entries:
        if e in entry_cart:
            coords.append(entry_cart[e])
    if len(coords) < 3:
        return
    coords = np.array(coords)
    try:
        tri = Delaunay(coords)
    except Exception:
        return

    drawn = set()
    first = True
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            if edge not in drawn:
                drawn.add(edge)
                p1, p2 = coords[edge[0]], coords[edge[1]]
                if first and label:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            label=label, zorder=zorder, **kwargs)
                    first = False
                else:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            zorder=zorder, **kwargs)


def load_pearson_from_db(db_path, structure_ids):
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
    raise ValueError(f"Unknown reference_phase_source in {dft_json}: '{src}'")


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


def filter_for_chemsys(entries, chemsys):
    elements = set(chemsys.split('-'))
    return [e for e in entries
            if set(str(el) for el in e.composition.elements).issubset(elements)]


def detect_ternary_systems(entries):
    systems = set()
    for e in entries:
        elems = sorted(str(el) for el in e.composition.elements)
        if len(elems) == 3:
            systems.add('-'.join(elems))
    return sorted(systems)


def plot_ternary_hull(chemsys, mp_entries_sys, gen_entries_sys, ax, e_hull_max,
                      near_threshold=0.01, pearson_map=None):
    """Plot one ternary hull on the given axis."""
    elements = sorted(chemsys.split('-'))

    if len(mp_entries_sys) < 3:
        ax.text(0.5, 0.4, f'{chemsys}\nInsufficient MP data',
                ha='center', va='center', fontsize=14)
        return

    elem_set = set(elements)
    mp_elems = set()
    for e in mp_entries_sys:
        if len(e.composition.elements) == 1:
            mp_elems.add(str(e.composition.elements[0]))
    if not elem_set.issubset(mp_elems):
        ax.text(0.5, 0.4, f'{chemsys}\nMissing elemental refs',
                ha='center', va='center', fontsize=14)
        return

    try:
        pd_mp = PhaseDiagram(mp_entries_sys)
    except Exception as exc:
        ax.text(0.5, 0.4, f'{chemsys}\nPD error: {str(exc)[:40]}',
                ha='center', va='center', fontsize=12)
        return

    all_entries = mp_entries_sys + gen_entries_sys
    try:
        pd_all = PhaseDiagram(all_entries)
    except Exception:
        pd_all = pd_mp

    if pearson_map is None:
        pearson_map = {}
    near_threshold = min(near_threshold, e_hull_max)

    def cart_of(entry):
        bary = np.array([entry.composition.get_atomic_fraction(el)
                         for el in elements])
        return bary2cart(bary)

    # Build entry -> cart mapping for hull facet drawing
    entry_cart = {}
    for e in set(list(pd_mp.stable_entries) + list(pd_all.stable_entries)):
        entry_cart[e] = cart_of(e)

    # Classify MP phases
    mp_on_any_hull = []
    for e in mp_entries_sys:
        on_hull = e in pd_all.stable_entries
        on_mp_hull = e in pd_mp.stable_entries
        if on_hull or on_mp_hull:
            mp_on_any_hull.append({'cart': cart_of(e), 'entry': e,
                                   'formula': e.composition.reduced_formula})

    # Deduplicate generated structures by formula (keep lowest energy per composition)
    gen_by_formula = {}
    for e in gen_entries_sys:
        if len(e.composition.elements) != 3:
            continue
        formula = e.composition.reduced_formula
        epa = e.energy / e.composition.num_atoms
        if formula not in gen_by_formula or epa < gen_by_formula[formula]['epa']:
            gen_by_formula[formula] = {'entry': e, 'epa': epa}

    # Classify deduplicated entries
    stable_gen = []
    near_meta_gen = []
    meta_gen = []
    for formula, item in gen_by_formula.items():
        e = item['entry']
        cart = cart_of(e)
        e_hull = pd_all.get_e_above_hull(e)
        rec = {'cart': cart, 'entry': e, 'e_hull': e_hull, 'epa': item['epa'],
               'formula': formula}
        if e in pd_all.stable_entries:
            stable_gen.append(rec)
        elif e_hull <= near_threshold:
            near_meta_gen.append(rec)
        elif e_hull <= e_hull_max:
            meta_gen.append(rec)

    # -- Colormap setup (rainbow: purple at E=0, red at E=max) --
    cmap = plt.cm.rainbow
    norm = plt.Normalize(vmin=0.0, vmax=e_hull_max)
    stable_purple = cmap(norm(0.0))

    # -- Draw ternary frame --
    draw_ternary_axes(ax, elements)

    # -- Hull facets: stable_purple solid actual hull, black dashed MP-only on top --
    draw_hull_facets(ax, pd_all, entry_cart, label='Convex hull', zorder=2,
                     color=stable_purple, ls='-', lw=2.5, alpha=0.8)
    draw_hull_facets(ax, pd_mp, entry_cart, label='MP-only hull', zorder=3,
                     color='black', ls='--', lw=2.0, alpha=0.7)

    # -- MP phases on either hull --
    mp_colors = plt.cm.tab20(np.linspace(0, 1, max(len(mp_on_any_hull), 1)))
    for i, d in enumerate(mp_on_any_hull):
        color = mp_colors[i % len(mp_colors)]
        ax.scatter(d['cart'][0], d['cart'][1], c=[color], s=150, marker='o',
                   edgecolors='black', linewidth=1.5, zorder=4,
                   label=formula_to_latex(d['formula']))

    # -- New stable structures (purple fill from colorbar E=0) --
    STABLE_MARKERS = ['^', 'D', 'v', 'p', 'P', 'X', 'h', 'H', '<', '>']
    for i, d in enumerate(stable_gen):
        marker = STABLE_MARKERS[i % len(STABLE_MARKERS)]
        ax.scatter(d['cart'][0], d['cart'][1], color=stable_purple, s=180,
                   marker=marker, edgecolors='red', linewidth=2.5, zorder=6)
        sid = d['entry'].name
        pearson = pearson_map.get(sid, '')
        formula_latex = formula_to_latex(d['formula'])
        label = f'{pearson}-{formula_latex}' if pearson else formula_latex
        ax.scatter([], [], color=stable_purple, s=180, marker=marker,
                   edgecolors='red', linewidth=2.5, label=label)

    # -- Near-metastable structures (within near_threshold): colormap fill + legend --
    for d in near_meta_gen:
        facecolor = cmap(norm(d['e_hull']))
        ax.scatter(d['cart'][0], d['cart'][1], c=[facecolor], s=150, marker='s',
                   edgecolors='red', linewidth=2, zorder=5)
        sid = d['entry'].name
        pearson = pearson_map.get(sid, '')
        formula_latex = formula_to_latex(d['formula'])
        label = f'{pearson}-{formula_latex}' if pearson else formula_latex
        ax.scatter([], [], c=[facecolor], s=150, marker='s',
                   edgecolors='red', linewidth=2, label=label)

    # -- Remaining metastable structures: colormap fill, no legend --
    for d in meta_gen:
        facecolor = cmap(norm(d['e_hull']))
        ax.scatter(d['cart'][0], d['cart'][1], c=[facecolor], s=120, marker='s',
                   edgecolors='red', linewidth=1.5, zorder=5)

    # -- Axes config --
    ax.set_aspect('equal')
    ax.axis('off')

    if meta_gen or near_meta_gen:
        title = f'{chemsys} ($E_{{\\mathrm{{hull}}}} \\leq {e_hull_max:.2f}$ eV/atom)'
    else:
        title = chemsys
    ax.set_title(title, fontsize=22, fontweight='bold', pad=10)
    ax.legend(fontsize=14, loc='upper left', framealpha=0.65,
              edgecolor='black', facecolor='white')

    # -- Colorbar for metastable structures --
    if meta_gen or near_meta_gen:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax, orientation='vertical',
                                  pad=0.02, shrink=0.85)
        cbar.set_label('Energy above hull (eV/atom)', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)


def main():
    parser = argparse.ArgumentParser(
        description='Plot ternary convex hulls for ternary magnet results')
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
                        help='Specific ternary chemsys to plot (default: auto-detect)')
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
        systems = detect_ternary_systems(gen_entries)

    print(f"  Ternary systems: {len(systems)} detected\n")

    if not systems:
        print("No ternary systems found.")
        return 0

    for chemsys in systems:
        mp_sub = filter_for_chemsys(mp_entries, chemsys)
        gen_sub = filter_for_chemsys(gen_entries, chemsys)
        print(f"  {chemsys}: {len(mp_sub)} MP-ref, {len(gen_sub)} generated")

        fig, ax = plt.subplots(figsize=(14, 10))
        plot_ternary_hull(chemsys, mp_sub, gen_sub, ax, args.e_above_hull_max,
                          near_threshold=args.near_threshold,
                          pearson_map=pearson_map)
        fig.subplots_adjust(left=0.03, right=1.00, bottom=0.01, top=0.95)

        stem = chemsys.replace('-', '')
        png_path = odir / f'{stem}_ternary_hull.png'
        pdf_path = odir / f'{stem}_ternary_hull.pdf'
        fig.savefig(str(png_path), dpi=300, bbox_inches='tight')
        fig.savefig(str(pdf_path), bbox_inches='tight')
        plt.close(fig)

        print(f"    -> {png_path}")
        print(f"    -> {pdf_path}")

    print(f"\nAll {len(systems)} ternary hull plots saved to: {odir.resolve()}")
    return 0


if __name__ == '__main__':
    exit(main())
