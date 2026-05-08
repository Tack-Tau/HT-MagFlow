#!/usr/bin/env python3
"""
Reproduce ft_eval_scatter.png locally from saved JSON evaluation results.

Reads baseline_eval_results.json and ft_eval_results.json, merges all splits
into a single scatter per model, and produces a 2-row (Original / Fine-tuned)
comparison plot.

Usage:
    python3 plot_eval_results.py --baseline baseline_eval_results.json \
                                 --finetuned ft_eval_results.json \
                                 --output ft_eval_scatter_local.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_all_frames(json_path):
    """Load per-frame data from eval results JSON, returning merged arrays."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    per_split = data.get('per_split', {})
    per_mp = data.get('per_mp_id', {})

    vasp_all, ms_all, outlier_all = [], [], []
    n_normal = 0
    n_outlier = 0
    mpid_maes_normal, mpid_maes_outlier = [], []

    for mp_id, info in per_mp.items():
        frames = info.get('frames', [])
        if not frames:
            continue
        is_out = info.get('is_outlier', False)
        for fr in frames:
            vasp_all.append(fr['vasp_epa'])
            ms_all.append(fr['ms_epa'])
            outlier_all.append(is_out)

        mae = info.get('mae_eV_per_atom', 0.0)
        if is_out:
            n_outlier += len(frames)
            mpid_maes_outlier.append(mae)
        else:
            n_normal += len(frames)
            mpid_maes_normal.append(mae)

    vasp_arr = np.array(vasp_all)
    ms_arr = np.array(ms_all)
    outlier_arr = np.array(outlier_all, dtype=bool)

    total_splits = list(per_split.values())
    overall_mae = np.mean([s['mae'] for s in total_splits if s.get('mae') is not None]) if total_splits else 0.0
    overall_rmse = np.mean([s['rmse'] for s in total_splits if s.get('rmse') is not None]) if total_splits else 0.0

    mae_normal = float(np.mean(mpid_maes_normal)) if mpid_maes_normal else None
    mae_outlier = float(np.mean(mpid_maes_outlier)) if mpid_maes_outlier else None

    return {
        'vasp_epa': vasp_arr,
        'ms_epa': ms_arr,
        'is_outlier': outlier_arr,
        'n_normal': n_normal,
        'n_outlier': n_outlier,
        'mpid_avg_mae': overall_mae,
        'mpid_avg_rmse': overall_rmse,
        'mpid_avg_mae_normal': mae_normal,
        'mpid_avg_mae_outlier': mae_outlier,
        'n_mp_ids': len([m for m in per_mp if per_mp[m].get('frames')]),
    }


def plot_scatter(ax, data, lo, hi, label):
    normal_mask = ~data['is_outlier']
    outlier_mask = data['is_outlier']

    lbl_n = f"Normal N={int(np.sum(normal_mask))}"
    if data['mpid_avg_mae_normal'] is not None:
        lbl_n += f"\nMAE={data['mpid_avg_mae_normal']:.4f}"
    ax.scatter(data['vasp_epa'][normal_mask], data['ms_epa'][normal_mask],
               c='#1f77b4', s=24, alpha=0.6, edgecolors='black',
               linewidth=0.3, label=lbl_n)

    if np.any(outlier_mask):
        lbl_o = f"Outlier N={int(np.sum(outlier_mask))}"
        if data['mpid_avg_mae_outlier'] is not None:
            lbl_o += f"\nMAE={data['mpid_avg_mae_outlier']:.4f}"
        ax.scatter(data['vasp_epa'][outlier_mask], data['ms_epa'][outlier_mask],
                   c='red', s=36, alpha=0.8, marker='x', linewidth=1.2,
                   label=lbl_o)

    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.5)
    ax.fill_between([lo, hi], [lo - 0.05, hi - 0.05], [lo + 0.05, hi + 0.05],
                    alpha=0.08, color='green')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('VASP DFT energy (eV/atom)', fontsize=14)
    ax.set_ylabel('MatterSim energy (eV/atom)', fontsize=14)
    ax.set_title(f'{label} '
                 f'(MAE={data["mpid_avg_mae"]:.4f}, '
                 f'RMSE={data["mpid_avg_rmse"]:.4f})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce ft_eval_scatter.png from saved JSON results"
    )
    parser.add_argument('--baseline', type=str, default='baseline_eval_results.json',
                        help="Baseline eval results JSON")
    parser.add_argument('--finetuned', type=str, default='ft_eval_results.json',
                        help="Fine-tuned eval results JSON")
    parser.add_argument('--output', type=str, default='ft_eval_scatter_local.png',
                        help="Output plot file (default: ft_eval_scatter_local.png)")
    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline}")
    baseline = load_all_frames(args.baseline)
    print(f"  {baseline['n_mp_ids']} MP-IDs, "
          f"{len(baseline['vasp_epa'])} frames "
          f"({baseline['n_normal']} normal, {baseline['n_outlier']} outlier)")

    print(f"Loading fine-tuned: {args.finetuned}")
    finetuned = load_all_frames(args.finetuned)
    print(f"  {finetuned['n_mp_ids']} MP-IDs, "
          f"{len(finetuned['vasp_epa'])} frames "
          f"({finetuned['n_normal']} normal, {finetuned['n_outlier']} outlier)")

    all_e = np.concatenate([baseline['vasp_epa'], baseline['ms_epa'],
                            finetuned['vasp_epa'], finetuned['ms_epa']])
    margin = (all_e.max() - all_e.min()) * 0.05
    lo, hi = all_e.min() - margin, all_e.max() + margin

    fig, axes = plt.subplots(2, 1, figsize=(7, 12), squeeze=False)
    plot_scatter(axes[0, 0], baseline, lo, hi, 'Original')
    plot_scatter(axes[1, 0], finetuned, lo, hi, 'Fine-tuned')
    fig.tight_layout()

    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {args.output}")


if __name__ == '__main__':
    main()
