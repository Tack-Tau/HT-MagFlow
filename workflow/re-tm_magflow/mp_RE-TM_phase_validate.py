#!/usr/bin/env python3
"""
Validate MP reference phase energies for binary RE-TM systems.

Queries the Materials Project REST API for E_hull of binary RE-TM phases
across lanthanide rare earths for specific prototypes separated by space group,
then generates line-dot plots to identify anomalous reference energies.

Usage:
    /Users/tonyspc/miniforge3/envs/pymatgen/bin/python3 mp_RE-TM_phase_validate.py
"""

import os
import sys
import json
import time
import requests

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

API_KEY = os.environ.get('MP_API_KEY')
if not API_KEY:
    print("ERROR: MP_API_KEY environment variable not set")
    sys.exit(1)

BASE_URL = "https://api.materialsproject.org"
HEADERS = {"X-API-KEY": API_KEY}

LANTHANIDES = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
               'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

PROTOTYPES = {
    'RETM2_227':  {'ratio': 2.0, 'spg': 227, 'pearson': 'cF24', 'formula': {'Fe': r'RFe$_2$',        'Co': r'RCo$_2$'}},
    'RETM5_191':  {'ratio': 5.0, 'spg': 191, 'pearson': 'hP6',  'formula': {'Fe': r'RFe$_5$',        'Co': r'RCo$_5$'}},
    'R2TM7_166':  {'ratio': 3.5, 'spg': 166, 'pearson': 'hR18', 'formula': {'Fe': r'R$_2$Fe$_7$',    'Co': r'R$_2$Co$_7$'}},
    'R2TM7_194':  {'ratio': 3.5, 'spg': 194, 'pearson': 'hP36', 'formula': {'Fe': r'R$_2$Fe$_7$',    'Co': r'R$_2$Co$_7$'}},
    'R2TM17_166': {'ratio': 8.5, 'spg': 166, 'pearson': 'hR19', 'formula': {'Fe': r'R$_2$Fe$_{17}$', 'Co': r'R$_2$Co$_{17}$'}},
    'R2TM17_194': {'ratio': 8.5, 'spg': 194, 'pearson': 'hP38', 'formula': {'Fe': r'R$_2$Fe$_{17}$', 'Co': r'R$_2$Co$_{17}$'}},
}

TM_ELEMENTS = ['Fe', 'Co']
CACHE_FILE = 'mp_RE-TM_phase_cache.json'


def _get_with_retry(url, headers, params, max_retries=3, delay=2.0):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            print(f"  WARNING: API returned {resp.status_code}, retry {attempt+1}")
        except requests.exceptions.RequestException as e:
            print(f"  WARNING: request failed ({e}), retry {attempt+1}")
        time.sleep(delay * (attempt + 1))
    return None


def alpha_to_mpid(alpha_id):
    prefix, code = alpha_id.rsplit('-', 1)
    num = 0
    for ch in code:
        num = num * 26 + (ord(ch) - ord('a'))
    return f"{prefix}-{num}"


def query_chemsys(chemsys):
    """Query thermo (GGA-only E_hull) and summary (space group) endpoints,
    return merged list of dicts with material_id, formula, composition, e_hull, spg.
    """
    thermo_url = f"{BASE_URL}/materials/thermo/"
    thermo_all = []
    offset = 0
    while True:
        params = {
            "chemsys": chemsys,
            "_fields": "material_id,formula_pretty,composition,energy_above_hull,thermo_type",
            "_limit": 100, "_skip": offset,
        }
        resp = _get_with_retry(thermo_url, HEADERS, params)
        if resp is None:
            break
        batch = resp.json().get('data', [])
        if not batch:
            break
        thermo_all.extend(batch)
        if len(batch) < 100:
            break
        offset += 100

    gga_map = {}
    for d in thermo_all:
        if d.get('thermo_type') == 'GGA_GGA+U':
            gga_map[d['material_id']] = {
                'e_hull': d.get('energy_above_hull'),
                'formula': d.get('formula_pretty', ''),
                'composition': d.get('composition', {}),
            }

    time.sleep(0.3)

    summary_url = f"{BASE_URL}/materials/summary/"
    summary_all = []
    offset = 0
    while True:
        params = {
            "chemsys": chemsys,
            "_fields": "material_id,symmetry",
            "_limit": 100, "_skip": offset,
        }
        resp = _get_with_retry(summary_url, HEADERS, params)
        if resp is None:
            break
        batch = resp.json().get('data', [])
        if not batch:
            break
        summary_all.extend(batch)
        if len(batch) < 100:
            break
        offset += 100

    spg_map = {d['material_id']: d.get('symmetry', {}).get('number', 0)
               for d in summary_all}

    time.sleep(0.3)

    results = []
    for mid, info in gga_map.items():
        results.append({
            'material_id': mid,
            'formula': info['formula'],
            'composition': info['composition'],
            'e_hull': info['e_hull'],
            'spg': spg_map.get(mid, 0),
        })
    return results


def match_prototype(materials, re_elem, tm_elem, proto_info):
    """Find best material matching prototype by composition ratio AND space group."""
    target_ratio = proto_info['ratio']
    target_spg = proto_info['spg']

    matched = []
    for mat in materials:
        comp = mat.get('composition', {})
        if isinstance(comp, str):
            from pymatgen.core import Composition
            comp = dict(Composition(comp).as_dict())

        n_re = comp.get(re_elem, 0)
        n_tm = comp.get(tm_elem, 0)
        if n_re == 0 or n_tm == 0:
            continue
        if len([e for e, v in comp.items() if v > 0]) != 2:
            continue
        ratio = n_tm / n_re
        if abs(ratio - target_ratio) > 0.1:
            continue
        if mat.get('spg', 0) != target_spg:
            continue

        matched.append(mat)

    if not matched:
        return None

    best = min(matched, key=lambda x: x.get('e_hull', 999))
    raw_id = best['material_id']
    mpid = alpha_to_mpid(raw_id) if raw_id.startswith('mp-') and raw_id[3:].isalpha() else raw_id
    return {
        'material_id': mpid,
        'formula': best.get('formula', ''),
        'e_hull': best.get('e_hull'),
        'spg': best.get('spg', 0),
    }


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return None


def save_cache(data):
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    output_dir = 'paper/figs'
    os.makedirs(output_dir, exist_ok=True)

    cached = load_cache()
    if cached:
        print(f"Loaded cached data from {CACHE_FILE}")
        all_data = cached
    else:
        print("Querying Materials Project REST API...")
        all_data = {}
        for tm in TM_ELEMENTS:
            all_data[tm] = {pk: {} for pk in PROTOTYPES}
            for re in LANTHANIDES:
                chemsys = '-'.join(sorted([re, tm]))
                print(f"  Querying {chemsys}...")
                materials = query_chemsys(chemsys)
                for proto_key, proto_info in PROTOTYPES.items():
                    result = match_prototype(materials, re, tm, proto_info)
                    if result is not None:
                        all_data[tm][proto_key][re] = result
                        print(f"    {proto_key}: E_hull = {result['e_hull']:.4f} "
                              f"(spg {result['spg']}, {result['material_id']})")

        save_cache(all_data)
        print(f"\nCached results to {CACHE_FILE}")

    # --- Plotting ---
    FS_TITLE = 18
    FS_LABEL = 15
    FS_TICK = 13
    FS_LEGEND = 12
    LW = 2.0
    MS = 11

    MARKERS = ['o', 's', 'D', '^', 'v', 'P']
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    proto_keys = list(PROTOTYPES.keys())
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    for col, tm in enumerate(TM_ELEMENTS):
        ax = axes[col]

        all_re_with_data = set()
        for pk in proto_keys:
            all_re_with_data.update(all_data[tm].get(pk, {}).keys())
        x_labels = [re for re in LANTHANIDES if re in all_re_with_data]
        x_pos = {re: i for i, re in enumerate(x_labels)}

        for pidx, pk in enumerate(proto_keys):
            pinfo = PROTOTYPES[pk]
            data = all_data[tm].get(pk, {})
            re_elems = [re for re in LANTHANIDES if re in data]
            if not re_elems:
                continue

            xs = [x_pos[re] for re in re_elems]
            ys = [data[re]['e_hull'] for re in re_elems]
            label = f"{pinfo['formula'][tm]} ({pinfo['pearson']}, {pinfo['spg']})"

            ax.plot(xs, ys, color=COLORS[pidx], marker=MARKERS[pidx],
                    linewidth=LW, markersize=MS, label=label,
                    zorder=5, alpha=0.85)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=FS_TICK)
        ax.set_xlabel('Lanthanide element', fontsize=FS_LABEL)
        ax.set_ylabel(r'$E_{\mathrm{hull}}$ (eV/atom)', fontsize=FS_LABEL)
        ax.set_title(f'R\u2013{tm}', fontsize=FS_TITLE, fontweight='bold')
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.legend(fontsize=FS_LEGEND, loc='best', framealpha=0.9)

    fig.tight_layout()
    outpath_pdf = os.path.join(output_dir, 'mp_RE-TM_phase_validation.pdf')
    outpath_png = os.path.join(output_dir, 'mp_RE-TM_phase_validation.png')
    fig.savefig(outpath_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(outpath_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {outpath_pdf} and {outpath_png}")

    # --- Summary ---
    print("\n=== Summary ===")
    for tm in TM_ELEMENTS:
        print(f"\nR-{tm} systems:")
        for pk in proto_keys:
            data = all_data[tm].get(pk, {})
            if not data:
                print(f"  {pk}: no phases found in MP")
                continue
            print(f"  {pk}:")
            for re in LANTHANIDES:
                if re in data:
                    flag = ""
                    if data[re]['e_hull'] > 0.05:
                        flag = " <-- ABOVE HULL"
                    elif data[re]['e_hull'] > 0:
                        flag = " <-- metastable"
                    print(f"    {re:3s}: E_hull = {data[re]['e_hull']:.4f} eV/atom  "
                          f"(spg {data[re]['spg']}, {data[re]['material_id']}){flag}")


if __name__ == '__main__':
    main()
