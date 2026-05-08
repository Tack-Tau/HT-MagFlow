#!/usr/bin/env python3
"""
Fine-tune MatterSim on VASP-relaxed MP phase structures.

Reads VASP relaxation results from mp_relax_workflow.json (or mp_vasp_comparison.json),
extracts structures with DFT energies/forces/stresses from vasprun.xml ionic trajectories,
builds a train/val dataset in extended XYZ format, and launches MatterSim finetuning.

All ionic steps from each VASP relaxation are included, giving the model
diverse geometries with non-trivial forces.

Train/val/test split is done by mp_id (not by frame) to prevent data leakage.
Default split ratio: 80/10/10.
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write
from ase.calculators.singlepoint import SinglePointCalculator


KBAR_TO_EVA3 = 0.1 / 160.21766208


def _atoms_from_ionic_step(structure, energy, forces, stress_kbar, mp_id, step_idx):
    """Convert one ionic step to ASE Atoms with SinglePointCalculator."""
    stress_eV_A3 = np.array(stress_kbar) * KBAR_TO_EVA3
    atoms = AseAtomsAdaptor().get_atoms(structure)
    stress_voigt = stress_eV_A3.flatten()[[0, 4, 8, 5, 2, 1]]
    calc = SinglePointCalculator(
        atoms,
        energy=float(energy),
        forces=np.array(forces, dtype=np.float64),
        stress=stress_voigt,
    )
    atoms.calc = calc
    atoms.info['mp_id'] = mp_id
    atoms.info['ionic_step'] = int(step_idx)
    return atoms


def _extract_from_vasprun(vr, mp_id, skip_first=0, max_force=10.0):
    """
    Extract Atoms from all ionic steps of a parsed Vasprun.

    Skips the first `skip_first` steps, steps where electronic SCF hit NELM,
    and steps with max |force| > max_force.
    """
    entries = []
    nelm = vr.parameters.get('NELM', 120)

    for step_idx, step in enumerate(vr.ionic_steps):
        if step_idx < skip_first:
            continue
        if len(step['electronic_steps']) >= nelm:
            continue
        forces = step['forces']
        if np.max(np.abs(forces)) > max_force:
            continue
        entries.append(_atoms_from_ionic_step(
            step['structure'], step['e_0_energy'],
            forces, step['stress'], mp_id, step_idx,
        ))
    return entries


def collect_structures_from_workflow_db(db_path, skip_first=0, max_force=10.0):
    """
    Collect structures from mp_relax_workflow.json.
    Reads vasprun.xml for each RELAX_DONE/RELAX_TMOUT entry.
    """
    with open(db_path, 'r') as f:
        db = json.load(f)

    entries = []
    skipped = 0

    for mp_id, sdata in db['structures'].items():
        if sdata['state'] not in ('RELAX_DONE', 'RELAX_TMOUT'):
            continue

        relax_dir = sdata.get('relax_dir')
        if not relax_dir:
            skipped += 1
            continue

        vasprun_path = Path(relax_dir) / 'vasprun.xml'
        if not vasprun_path.exists():
            skipped += 1
            continue

        try:
            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
            frames = _extract_from_vasprun(vr, mp_id, skip_first, max_force)
            if frames:
                for a in frames:
                    a.info['is_outlier'] = False
                entries.extend(frames)
            else:
                skipped += 1

        except Exception as e:
            print(f"  Error processing {mp_id}: {e}")
            skipped += 1

    n_mp = len(set(a.info['mp_id'] for a in entries))
    print(f"Collected {len(entries)} frames from {n_mp} structures ({skipped} skipped)")
    return entries


def collect_structures_from_comparison(comparison_path, workflow_db_path,
                                       skip_first=0, max_force=10.0):
    """
    Collect structures using mp_vasp_comparison.json for validated mp_ids,
    then read vasprun.xml from relax_dir in workflow DB.

    Tags each frame's atoms.info['is_outlier'] from the comparison JSON.
    """
    with open(comparison_path, 'r') as f:
        comparison = json.load(f)

    with open(workflow_db_path, 'r') as f:
        db = json.load(f)

    outlier_lookup = {}
    for entry in comparison.get('structures', []):
        outlier_lookup[entry['mp_id']] = bool(entry.get('is_outlier', False))

    entries = []
    skipped = 0

    for mp_id in sorted(outlier_lookup.keys()):
        sdata = db['structures'].get(mp_id)
        if not sdata:
            skipped += 1
            continue

        relax_dir = sdata.get('relax_dir')
        if not relax_dir:
            skipped += 1
            continue

        vasprun_path = Path(relax_dir) / 'vasprun.xml'
        if not vasprun_path.exists():
            skipped += 1
            continue

        try:
            vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
            frames = _extract_from_vasprun(vr, mp_id, skip_first, max_force)
            if frames:
                is_out = outlier_lookup.get(mp_id, False)
                for a in frames:
                    a.info['is_outlier'] = is_out
                entries.extend(frames)
            else:
                skipped += 1

        except Exception as e:
            print(f"  Error processing {mp_id}: {e}")
            skipped += 1

    n_mp = len(set(a.info['mp_id'] for a in entries))
    n_out = len(set(a.info['mp_id'] for a in entries if a.info.get('is_outlier')))
    print(f"Collected {len(entries)} frames from {n_mp} structures "
          f"({n_out} outlier, {skipped} skipped)")
    return entries


def subsample_per_mp_id(entries, max_frames):
    """Cap frames per MP-ID by selecting evenly spaced ionic steps.

    Always includes first and last frame. When a trajectory exceeds
    max_frames, selects evenly spaced frames in between.
    """
    if max_frames is None or max_frames <= 0:
        return entries

    mp_id_to_atoms = {}
    for a in entries:
        mp_id_to_atoms.setdefault(a.info['mp_id'], []).append(a)

    result = []
    n_capped = 0
    for mp_id in sorted(mp_id_to_atoms.keys()):
        frames = mp_id_to_atoms[mp_id]
        frames.sort(key=lambda a: a.info['ionic_step'])

        if len(frames) <= max_frames:
            result.extend(frames)
        else:
            n_capped += 1
            indices = np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(int)
            indices = sorted(set(indices))
            result.extend(frames[i] for i in indices)

    if n_capped > 0:
        print(f"  Capped frames for {n_capped}/{len(mp_id_to_atoms)} structures "
              f"(max {max_frames} per MP-ID)")
    print(f"  After subsampling: {len(result)} frames "
          f"(was {len(entries)})")
    return result


def _split_ids(mp_ids, val_fraction, test_fraction, rng):
    """Shuffle and split a list of mp_ids into train/val/test."""
    ids = list(mp_ids)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_fraction))
    n_val = max(1, int(len(ids) * val_fraction))
    return ids[n_test + n_val:], ids[n_test:n_test + n_val], ids[:n_test]


def train_val_test_split(entries, val_fraction=0.1, test_fraction=0.1, seed=42):
    """
    Stratified split by mp_id: outlier and normal structures are split
    independently so each split preserves the outlier/normal ratio.
    All frames from the same mp_id go into the same split.
    """
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    mp_id_to_atoms = {}
    for a in entries:
        mp_id_to_atoms.setdefault(a.info['mp_id'], []).append(a)

    normal_ids = sorted(mid for mid in mp_id_to_atoms
                        if not mp_id_to_atoms[mid][0].info.get('is_outlier', False))
    outlier_ids = sorted(mid for mid in mp_id_to_atoms
                         if mp_id_to_atoms[mid][0].info.get('is_outlier', False))

    rng = np.random.default_rng(seed)

    train_n, val_n, test_n = _split_ids(normal_ids, val_fraction, test_fraction, rng)
    if outlier_ids:
        train_o, val_o, test_o = _split_ids(outlier_ids, val_fraction, test_fraction, rng)
    else:
        train_o, val_o, test_o = [], [], []

    train_ids = train_n + train_o
    val_ids = val_n + val_o
    test_ids = test_n + test_o

    train = [a for mid in train_ids for a in mp_id_to_atoms[mid]]
    val = [a for mid in val_ids for a in mp_id_to_atoms[mid]]
    test = [a for mid in test_ids for a in mp_id_to_atoms[mid]]

    if outlier_ids:
        print(f"  Stratified split: {len(train_n)}+{len(train_o)} train, "
              f"{len(val_n)}+{len(val_o)} val, "
              f"{len(test_n)}+{len(test_o)} test (normal+outlier structures)")

    return train, val, test


def oversample_outliers(entries, outlier_weight=1.0):
    """Oversample outlier frames in training set to approximate loss weighting.

    MatterSim's training loop uses uniform sampling, so repeating outlier
    frames achieves an effective per-sample weight proportional to repeat count.

    oversample factor = round(outlier_weight * N_normal_ids / N_outlier_ids)
    outlier_weight=1.0 gives balanced (equal total contribution from each group).
    """
    normal = [a for a in entries if not a.info.get('is_outlier', False)]
    outliers = [a for a in entries if a.info.get('is_outlier', False)]

    if not outliers:
        return entries

    n_normal_ids = len(set(a.info['mp_id'] for a in normal))
    n_outlier_ids = len(set(a.info['mp_id'] for a in outliers))

    factor = max(1, round(outlier_weight * n_normal_ids / n_outlier_ids))

    if factor <= 1:
        return entries

    result = list(normal)
    for _ in range(factor):
        result.extend(outliers)

    print(f"  Oversampled outlier frames {factor}x: "
          f"{len(outliers)} -> {len(outliers) * factor} "
          f"(total train: {len(result)} frames)")
    return result


def write_xyz_dataset(atoms_list, output_path):
    """Write list of ASE Atoms to extended XYZ file."""
    ase_write(str(output_path), atoms_list, format='extxyz')
    print(f"  Written {len(atoms_list)} structures to {output_path}")


def evaluate_model(model_path, xyz_paths, device='cuda'):
    """Single-point energy evaluation of finetuned model on train/val/test XYZ.

    Returns {split: {vasp_epa, ms_epa, mp_ids, is_outlier, mae, rmse,
                      mae_normal, rmse_normal, mae_outlier, rmse_outlier}}.
    Deduplplicates oversampled frames by (mp_id, ionic_step) before metrics.
    """
    from mattersim.forcefield import MatterSimCalculator
    from ase.io import read as ase_read

    print(f"\nLoading finetuned model: {model_path}")
    calc = MatterSimCalculator(load_path=str(model_path), device=device)

    results = {}
    for split, xyz_path in xyz_paths.items():
        xyz_path = Path(xyz_path)
        if not xyz_path.exists():
            print(f"  WARNING: {xyz_path} not found, skipping {split}")
            continue

        atoms_list = ase_read(str(xyz_path), index=':')

        vasp_epa, ms_epa, mp_ids, ionic_steps, is_outlier_flags = \
            [], [], [], [], []
        seen = set()

        print(f"  {split}: {len(atoms_list)} frames ...", end='', flush=True)
        for atoms in atoms_list:
            key = (atoms.info.get('mp_id', ''), atoms.info.get('ionic_step', 0))
            if key in seen:
                continue
            seen.add(key)

            n = len(atoms)
            vasp_epa.append(atoms.get_potential_energy() / n)
            atoms_eval = atoms.copy()
            atoms_eval.calc = calc
            ms_epa.append(atoms_eval.get_potential_energy() / n)
            mp_ids.append(atoms.info.get('mp_id', ''))
            ionic_steps.append(int(atoms.info.get('ionic_step', 0)))
            is_outlier_flags.append(bool(atoms.info.get('is_outlier', False)))

        vasp_epa = np.array(vasp_epa)
        ms_epa = np.array(ms_epa)
        is_outlier_arr = np.array(is_outlier_flags)
        mp_ids_arr = np.array(mp_ids)
        diff = ms_epa - vasp_epa
        abs_diff = np.abs(diff)

        # Per-MP-ID MAE/RMSE, then Q3+3*IQR to flag bad predictions
        unique_ids = sorted(set(mp_ids))
        mpid_mae = {}
        mpid_rmse = {}
        for mid in unique_ids:
            mask = mp_ids_arr == mid
            mpid_mae[mid] = float(np.mean(abs_diff[mask]))
            mpid_rmse[mid] = float(np.sqrt(np.mean(diff[mask] ** 2)))

        mae_vals = np.array(list(mpid_mae.values()))
        q1 = np.percentile(mae_vals, 25)
        q3 = np.percentile(mae_vals, 75)
        bad_thresh = q3 + 3 * (q3 - q1)
        bad_mp_ids = set(mid for mid, v in mpid_mae.items() if v > bad_thresh)

        is_bad_pred = np.array([mid in bad_mp_ids for mid in mp_ids])

        mpid_avg_mae = float(np.mean(mae_vals))
        mpid_avg_rmse = float(np.mean(list(mpid_rmse.values())))

        # Per-group (normal/outlier) MP-ID averaged metrics
        outlier_id_set = set(mp_ids_arr[is_outlier_arr])
        normal_ids = [mid for mid in unique_ids if mid not in outlier_id_set]
        outlier_ids_list = [mid for mid in unique_ids if mid in outlier_id_set]

        split_result = {
            'vasp_epa': vasp_epa,
            'ms_epa': ms_epa,
            'mp_ids': mp_ids,
            'ionic_steps': ionic_steps,
            'is_outlier': is_outlier_flags,
            'is_bad_pred': is_bad_pred,
            'mpid_avg_mae': mpid_avg_mae,
            'mpid_avg_rmse': mpid_avg_rmse,
            'bad_pred_threshold': float(bad_thresh),
            'bad_pred_mp_ids': sorted(bad_mp_ids),
            'mpid_mae': mpid_mae,
        }

        msg = f" MAE={mpid_avg_mae:.4f}, RMSE={mpid_avg_rmse:.4f} eV/atom"
        if normal_ids:
            mpid_avg_mae_n = float(np.mean([mpid_mae[m] for m in normal_ids]))
            split_result['mpid_avg_mae_normal'] = mpid_avg_mae_n
            msg += f" | normal: {mpid_avg_mae_n:.4f}"
        if outlier_ids_list:
            mpid_avg_mae_o = float(np.mean([mpid_mae[m] for m in outlier_ids_list]))
            split_result['mpid_avg_mae_outlier'] = mpid_avg_mae_o
            msg += f" | outlier: {mpid_avg_mae_o:.4f}"

        n_unique = len(seen)
        n_bad = len(bad_mp_ids)
        print(f" ({n_unique} unique){msg}")
        if n_bad > 0:
            print(f"    Bad predictions (per-MP MAE > {bad_thresh:.4f}): "
                  f"{n_bad} MP-IDs")
            for mid in sorted(bad_mp_ids):
                tag = " [outlier]" if mpid_mae.get(mid) and \
                    any(is_outlier_flags[i] for i, m in enumerate(mp_ids)
                        if m == mid) else ""
                print(f"      {mid}: MAE={mpid_mae[mid]:.4f}{tag}")

        results[split] = split_result

    return results


def _plot_eval_row(axes, results, splits, lo, hi, row_label):
    """Plot one row of scatter subplots for a single model."""
    for idx, split in enumerate(splits):
        ax = axes[idx]
        d = results[split]
        n_frames = len(d['vasp_epa'])
        is_out = np.array(d.get('is_outlier', [False] * n_frames))
        normal_mask = ~is_out
        outlier_mask = is_out

        label_n = f"Normal N={int(np.sum(normal_mask))}"
        if 'mpid_avg_mae_normal' in d:
            label_n += f"\nMAE={d['mpid_avg_mae_normal']:.4f}"
        ax.scatter(d['vasp_epa'][normal_mask], d['ms_epa'][normal_mask],
                   c='#1f77b4', s=12, alpha=0.6, edgecolors='black',
                   linewidth=0.3, label=label_n)

        if np.any(outlier_mask):
            label_o = f"Outlier N={int(np.sum(outlier_mask))}"
            if 'mpid_avg_mae_outlier' in d:
                label_o += f"\nMAE={d['mpid_avg_mae_outlier']:.4f}"
            ax.scatter(d['vasp_epa'][outlier_mask], d['ms_epa'][outlier_mask],
                       c='red', s=20, alpha=0.8, marker='x', linewidth=1.0,
                       label=label_o)

        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, alpha=0.5)
        ax.fill_between([lo, hi], [lo - 0.05, hi - 0.05], [lo + 0.05, hi + 0.05],
                        alpha=0.08, color='green')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel('VASP DFT energy (eV/atom)', fontsize=10)
        ax.set_ylabel('MatterSim energy (eV/atom)', fontsize=10)
        ax.set_title(f'{row_label} - {split.capitalize()} '
                     f'(MAE={d["mpid_avg_mae"]:.4f}, '
                     f'RMSE={d["mpid_avg_rmse"]:.4f})',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)


def plot_evaluation(results, output_path, baseline_results=None):
    """Scatter plot of VASP vs MatterSim energy per atom.

    If baseline_results is provided, generates a 2-row comparison plot
    (top: original pretrained, bottom: fine-tuned) with shared axis range.
    Otherwise generates a single-row plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if baseline_results is not None:
        splits = [s for s in ('train', 'val', 'test')
                  if s in results and s in baseline_results]
    else:
        splits = [s for s in ('train', 'val', 'test') if s in results]
    if not splits:
        return

    all_arrays = [results[s]['vasp_epa'] for s in splits] + \
                 [results[s]['ms_epa'] for s in splits]
    if baseline_results is not None:
        all_arrays += [baseline_results[s]['vasp_epa'] for s in splits] + \
                      [baseline_results[s]['ms_epa'] for s in splits]
    all_e = np.concatenate(all_arrays)
    margin = (all_e.max() - all_e.min()) * 0.05
    lo, hi = all_e.min() - margin, all_e.max() + margin

    n = len(splits)
    if baseline_results is not None:
        fig, axes = plt.subplots(2, n, figsize=(6 * n, 11), squeeze=False)
        _plot_eval_row(axes[0], baseline_results, splits, lo, hi, 'Original')
        _plot_eval_row(axes[1], results, splits, lo, hi, 'Fine-tuned')
        fig.suptitle('Original vs Fine-tuned MatterSim (Single-Point Energy)',
                     fontsize=14, fontweight='bold')
    else:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
        _plot_eval_row(axes[0], results, splits, lo, hi, 'Fine-tuned')
        fig.suptitle('Fine-tuned MatterSim vs VASP DFT (Single-Point Energy)',
                     fontsize=13, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Evaluation scatter plot: {output_path}")


def save_eval_results(results, output_path):
    """Save per-mp_id and per-split evaluation statistics to JSON."""
    out = {'per_split': {}, 'per_mp_id': {}}

    for split, d in results.items():
        split_info = {
            'n_frames': int(len(d['vasp_epa'])),
            'n_mp_ids': len(set(d['mp_ids'])),
            'mae': d.get('mpid_avg_mae'),
            'rmse': d.get('mpid_avg_rmse'),
        }
        for k in ('mpid_avg_mae_normal', 'mpid_avg_mae_outlier',
                   'bad_pred_threshold'):
            if k in d:
                split_info[k] = d[k]
        if 'bad_pred_mp_ids' in d:
            split_info['n_bad_pred'] = len(d['bad_pred_mp_ids'])
            split_info['bad_pred_mp_ids'] = d['bad_pred_mp_ids']
        out['per_split'][split] = split_info

        bad_set = set(d.get('bad_pred_mp_ids', []))
        is_outlier_map = {}
        mp_data = {}
        for i, mp_id in enumerate(d['mp_ids']):
            step = d['ionic_steps'][i] if 'ionic_steps' in d else i
            mp_data.setdefault(mp_id, {'vasp': [], 'ms': [], 'steps': []})
            mp_data[mp_id]['vasp'].append(float(d['vasp_epa'][i]))
            mp_data[mp_id]['ms'].append(float(d['ms_epa'][i]))
            mp_data[mp_id]['steps'].append(int(step))
            if 'is_outlier' in d:
                is_outlier_map[mp_id] = d['is_outlier'][i]

        for mp_id, md in sorted(mp_data.items()):
            vasp_arr = np.array(md['vasp'])
            ms_arr = np.array(md['ms'])
            diff = ms_arr - vasp_arr
            entry = {
                'split': split,
                'is_outlier': is_outlier_map.get(mp_id, False),
                'is_bad_pred': mp_id in bad_set,
                'n_frames': len(diff),
                'mae_eV_per_atom': float(np.mean(np.abs(diff))),
                'rmse_eV_per_atom': float(np.sqrt(np.mean(diff ** 2))),
                'mean_vasp_epa': float(np.mean(vasp_arr)),
                'mean_ms_epa': float(np.mean(ms_arr)),
            }
            entry['frames'] = [
                {'ionic_step': md['steps'][j],
                 'vasp_epa': md['vasp'][j],
                 'ms_epa': md['ms'][j],
                 'error': float(diff[j])}
                for j in range(len(diff))
            ]
            if mp_id in out['per_mp_id']:
                out['per_mp_id'][mp_id].update(entry)
            else:
                out['per_mp_id'][mp_id] = entry

    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Evaluation results JSON: {output_path}")


def run_finetune(train_path, val_path, save_path, model_name, epochs,
                 batch_size, lr, device, include_stresses,
                 patience, re_normalize):
    """Launch MatterSim finetuning via torchrun."""
    cmd = [
        'torchrun', '--nproc_per_node=1',
        '-m', 'mattersim.training.finetune_mattersim',
        '--load_model_path', model_name,
        '--train_data_path', str(train_path),
        '--valid_data_path', str(val_path),
        '--save_path', str(save_path),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--device', device,
        '--save_checkpoint',
        '--ckpt_interval', '20',
        '--early_stop_patience', str(patience),
        '--include_forces',
    ]

    if include_stresses:
        cmd.append('--include_stresses')
    if re_normalize:
        cmd.append('--re_normalize')

    print(f"\nFinetuning command:")
    print(f"  {' '.join(cmd)}\n")
    sys.stdout.flush()

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MatterSim on VASP-relaxed MP phase structures"
    )
    parser.add_argument(
        '--workflow-db',
        type=str,
        default='./mp_relax_workflow.json',
        help="Path to mp_relax_workflow.json"
    )
    parser.add_argument(
        '--comparison-json',
        type=str,
        default=None,
        help="Path to mp_vasp_comparison.json (uses validated structures only)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ft_mattersim',
        help="Output directory for datasets and finetuned model"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mattersim-v1.0.0-5m',
        choices=['mattersim-v1.0.0-1m', 'mattersim-v1.0.0-5m'],
        help="Pre-trained MatterSim model to finetune"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help="Training batch size (small for <200 structures)"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        '--val-fraction',
        type=float,
        default=0.1,
        help="Fraction of structures for validation (default: 0.1)"
    )
    parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.1,
        help="Fraction of structures for test set (default: 0.1)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for train/val split"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="Device for training (cuda or cpu)"
    )
    parser.add_argument(
        '--include-stresses',
        action='store_true',
        default=True,
        help="Include stress tensor in training (default: True)"
    )
    parser.add_argument(
        '--no-stresses',
        action='store_true',
        help="Disable stress tensor in training"
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help="Early stopping patience in epochs (default: 50)"
    )
    parser.add_argument(
        '--re-normalize',
        action='store_true',
        help="Re-normalize energy/force scales to the finetuning data"
    )
    parser.add_argument(
        '--dataset-only',
        action='store_true',
        help="Only prepare dataset (train/val/test XYZ files), do not launch training"
    )
    parser.add_argument(
        '--skip-first',
        type=int,
        default=0,
        help="Skip the first N ionic steps per trajectory (default: 0)"
    )
    parser.add_argument(
        '--max-force',
        type=float,
        default=10.0,
        help="Skip ionic steps with max |force| > this value in eV/A (default: 10.0)"
    )
    parser.add_argument(
        '--max-frames-per-id',
        type=int,
        default=None,
        help="Cap frames per MP-ID by evenly sampling (default: None = keep all)"
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help="Only evaluate an existing finetuned model (skip data prep and training)"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to finetuned model checkpoint for evaluation"
    )
    parser.add_argument(
        '--outlier-weight',
        type=float,
        default=1.0,
        help=("Outlier oversampling weight (default: 1.0 = balanced). "
              "Oversample factor = round(weight * N_normal / N_outlier). "
              "Only applied to training set.")
    )

    args = parser.parse_args()

    if args.no_stresses:
        args.include_stresses = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --eval-only: skip data prep and training, just evaluate
    if args.eval_only:
        model_path = args.model_path
        if not model_path:
            pth_files = sorted((output_dir / 'model').glob('*_ft.pth'))
            if not pth_files:
                pth_files = sorted((output_dir / 'model').glob('*.pth'))
            if pth_files:
                model_path = str(pth_files[-1])
            else:
                print("ERROR: No model found. Use --model-path.")
                return 1

        xyz_paths = {
            'train': output_dir / 'train.xyz',
            'val': output_dir / 'val.xyz',
            'test': output_dir / 'test.xyz',
        }
        print("=" * 70)
        print("Evaluation Only Mode")
        print("=" * 70)

        print("\n--- Original (pretrained) model ---")
        baseline_results = evaluate_model(args.model, xyz_paths, args.device)
        save_eval_results(baseline_results,
                          output_dir / 'baseline_eval_results.json')

        print("\n--- Fine-tuned model ---")
        results = evaluate_model(model_path, xyz_paths, args.device)
        save_eval_results(results, output_dir / 'ft_eval_results.json')

        plot_evaluation(results, output_dir / 'ft_eval_scatter.png',
                        baseline_results=baseline_results)
        return 0

    print("=" * 70)
    print("MatterSim Fine-tuning on VASP-relaxed MP Phases")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    train_frac = 1.0 - args.val_fraction - args.test_fraction
    print(f"Split: {train_frac:.0%} train / {args.val_fraction:.0%} val / {args.test_fraction:.0%} test")
    print(f"Include stresses: {args.include_stresses}")
    print(f"Early stop patience: {args.patience}")
    print(f"Re-normalize: {args.re_normalize}")
    print(f"Device: {args.device}")
    print(f"Trajectory filter: skip_first={args.skip_first}, max_force={args.max_force}")
    print(f"Max frames per ID: {args.max_frames_per_id or 'all'}")
    print(f"Outlier weight: {args.outlier_weight}")
    print("=" * 70 + "\n")

    traj_kw = dict(skip_first=args.skip_first, max_force=args.max_force)

    # Collect structures
    if args.comparison_json and Path(args.comparison_json).exists():
        print(f"Using validated structures from: {args.comparison_json}")
        entries = collect_structures_from_comparison(
            args.comparison_json, args.workflow_db, **traj_kw
        )
    else:
        print(f"Reading structures from workflow DB: {args.workflow_db}")
        entries = collect_structures_from_workflow_db(args.workflow_db, **traj_kw)

    if len(entries) == 0:
        print("ERROR: No valid structures collected. Exiting.")
        return 1

    n_mp = len(set(a.info['mp_id'] for a in entries))
    print(f"\nTotal frames: {len(entries)} from {n_mp} structures")

    if args.max_frames_per_id is not None:
        entries = subsample_per_mp_id(entries, args.max_frames_per_id)

    # Train/val/test split (by mp_id to prevent data leakage)
    train_set, val_set, test_set = train_val_test_split(
        entries, args.val_fraction, args.test_fraction, args.seed
    )
    train_ids = sorted(set(a.info['mp_id'] for a in train_set))
    val_ids = sorted(set(a.info['mp_id'] for a in val_set))
    test_ids = sorted(set(a.info['mp_id'] for a in test_set))
    print(f"Train: {len(train_set)} frames ({len(train_ids)} structures)")
    print(f"Val:   {len(val_set)} frames ({len(val_ids)} structures)")
    print(f"Test:  {len(test_set)} frames ({len(test_ids)} structures)")

    # Oversample outlier frames in training set only
    if args.outlier_weight > 0:
        train_set = oversample_outliers(train_set, args.outlier_weight)

    # Write XYZ datasets
    train_path = output_dir / 'train.xyz'
    val_path = output_dir / 'val.xyz'
    test_path = output_dir / 'test.xyz'
    write_xyz_dataset(train_set, train_path)
    write_xyz_dataset(val_set, val_path)
    write_xyz_dataset(test_set, test_path)

    # Save split metadata
    outlier_mp_ids = set(a.info['mp_id'] for a in entries if a.info.get('is_outlier'))
    n_out_train = len(set(train_ids) & outlier_mp_ids)
    n_out_val = len(set(val_ids) & outlier_mp_ids)
    n_out_test = len(set(test_ids) & outlier_mp_ids)
    meta = {
        'timestamp': datetime.now().isoformat(),
        'workflow_db': str(args.workflow_db),
        'comparison_json': str(args.comparison_json) if args.comparison_json else None,
        'skip_first': args.skip_first,
        'max_force': args.max_force,
        'outlier_weight': args.outlier_weight,
        'n_structures': n_mp,
        'n_total_frames': len(entries),
        'n_train_frames': len(train_set),
        'n_val_frames': len(val_set),
        'n_test_frames': len(test_set),
        'n_train_structures': len(train_ids),
        'n_val_structures': len(val_ids),
        'n_test_structures': len(test_ids),
        'n_outlier_train': n_out_train,
        'n_outlier_val': n_out_val,
        'n_outlier_test': n_out_test,
        'val_fraction': args.val_fraction,
        'test_fraction': args.test_fraction,
        'seed': args.seed,
        'model': args.model,
        'train_mp_ids': train_ids,
        'val_mp_ids': val_ids,
        'test_mp_ids': test_ids,
    }
    meta_path = output_dir / 'dataset_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Dataset metadata saved to {meta_path}")

    if args.dataset_only:
        print("\n--dataset-only: skipping training.")
        return 0

    # Run finetuning
    print("\n" + "=" * 70)
    print("Starting MatterSim Finetuning")
    print("=" * 70)
    sys.stdout.flush()

    save_path = output_dir / 'model'
    exit_code = run_finetune(
        train_path=train_path,
        val_path=val_path,
        save_path=save_path,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        include_stresses=args.include_stresses,
        patience=args.patience,
        re_normalize=args.re_normalize,
    )

    if exit_code == 0:
        best_src = save_path / 'best_model.pth'
        model_ckpt = best_src
        if best_src.exists():
            date_str = datetime.now().strftime('%Y%m%d')
            base_name = args.model.replace('.pth', '')
            ft_name = f"{base_name}_{date_str}_ft.pth"
            ft_dst = save_path / ft_name
            best_src.rename(ft_dst)
            print(f"\nRenamed best_model.pth -> {ft_name}")
            model_ckpt = ft_dst

        print("\n" + "=" * 70)
        print("Finetuning completed successfully!")
        print(f"Model saved to: {model_ckpt}")
        print("=" * 70)

        # Post-finetuning evaluation
        if model_ckpt.exists():
            print("\n" + "=" * 70)
            print("Post-Finetuning Evaluation (Single-Point Energy)")
            print("=" * 70)
            try:
                xyz_paths = {
                    'train': train_path,
                    'val': val_path,
                    'test': test_path,
                }
                print("\n--- Original (pretrained) model ---")
                baseline_results = evaluate_model(
                    args.model, xyz_paths, args.device)
                save_eval_results(baseline_results,
                                  output_dir / 'baseline_eval_results.json')

                print("\n--- Fine-tuned model ---")
                eval_results = evaluate_model(model_ckpt, xyz_paths, args.device)
                save_eval_results(eval_results, output_dir / 'ft_eval_results.json')

                plot_evaluation(eval_results, output_dir / 'ft_eval_scatter.png',
                                baseline_results=baseline_results)
            except Exception as e:
                print(f"\nWARNING: Post-finetuning evaluation failed: {e}")
    else:
        print(f"\nERROR: Finetuning failed with exit code {exit_code}")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
