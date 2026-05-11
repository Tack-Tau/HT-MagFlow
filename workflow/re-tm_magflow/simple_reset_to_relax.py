#!/usr/bin/env python3
"""
Reset jobs to RELAX_FAILED regardless of current state.

For magnet workflow (bin_mag_flow.py / ter_mag_flow.py) which only has
RELAX-stage states: PENDING, RELAX_RUNNING, RELAX_DONE, RELAX_TMOUT, RELAX_FAILED.

This script forces selected structures back to RELAX_FAILED so that
reset_failed_jobs.py can then reset them to PENDING for a full retry.

Usage:
    python3 simple_reset_to_relax.py --workflow VASP_JOBS/workflow.json --chemsys "Co-Sm-*" --dry-run
    python3 simple_reset_to_relax.py --workflow VASP_JOBS/workflow.json --chemsys "Co-Sm-*" --clean
    python3 simple_reset_to_relax.py --workflow VASP_JOBS/workflow.json --chemsys "Co-Sm-*" "Co-Tb-*" --clean
"""

import json
import sys
import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path


def parse_chemsys_pattern(chemsys_input):
    """
    Parse a chemsys input into (fixed_elements, is_wildcard).

    Order-independent: "Co-Sm-*", "*-Sm-Co", "Sm-*-Co" are all equivalent.
    Exact: "Al-Ca-S", "Ca-S-Al" are equivalent (sorted to "Al-Ca-S").
    """
    parts = [p.strip() for p in chemsys_input.split('-')]
    fixed = sorted([p for p in parts if p != '*'])
    is_wildcard = '*' in parts
    return fixed, is_wildcard


def chemsys_matches(stored_chemsys, fixed_elements, is_wildcard):
    """Check if a stored chemsys matches a single pattern."""
    stored_els = stored_chemsys.split('-')
    if is_wildcard:
        return all(el in stored_els for el in fixed_elements)
    else:
        return stored_els == fixed_elements


def chemsys_matches_any(stored_chemsys, patterns):
    """Check if stored chemsys matches any of the parsed patterns."""
    for fixed_els, is_wc in patterns:
        if chemsys_matches(stored_chemsys, fixed_els, is_wc):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Reset jobs to RELAX_FAILED regardless of current state (magnet workflow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--workflow',
        default='VASP_JOBS/workflow.json',
        help='Path to workflow.json (default: VASP_JOBS/workflow.json)'
    )
    parser.add_argument(
        '--chemsys',
        nargs='+',
        required=True,
        help="Filter by chemical system (order-independent, multiple patterns allowed). "
             "Exact: 'Co-Cr-Gd'. Wildcard: 'Co-Gd-*' (any system with Co and Gd). "
             "Example: --chemsys 'Co-Sm-*' 'Co-Tb-*'"
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help="Remove Relax directories for matched structures. "
             "Without this flag, only workflow.json states are reset."
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without executing'
    )

    args = parser.parse_args()

    if not os.path.exists(args.workflow):
        print(f"ERROR: Workflow database not found: {args.workflow}")
        sys.exit(1)

    patterns = [parse_chemsys_pattern(c) for c in args.chemsys]

    print("=" * 67)
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print("=" * 67)
    print("Simple Reset to RELAX_FAILED (magnet workflow)")
    print("=" * 67)
    print(f"Workflow database: {args.workflow}")
    print(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"Chemical system filter: {' '.join(args.chemsys)}")
    for c, (fixed_els, is_wc) in zip(args.chemsys, patterns):
        if is_wc:
            print(f"  {c} -> wildcard, matching systems containing {fixed_els}")
        else:
            print(f"  {c} -> exact, matching system {'-'.join(fixed_els)}")
    print()

    with open(args.workflow, 'r') as f:
        data = json.load(f)

    # Find structures matching chemsys that are NOT already RELAX_FAILED or PENDING
    skip_states = {'RELAX_FAILED', 'PENDING'}
    matched = []
    for struct_id, sdata in data['structures'].items():
        stored = sdata.get('chemsys', '')
        if not chemsys_matches_any(stored, patterns):
            continue
        if sdata['state'] in skip_states:
            continue
        matched.append(struct_id)

    if not matched:
        print("No structures to reset (all matching structures are already PENDING or RELAX_FAILED).")
        sys.exit(0)

    # Group by current state for reporting
    by_state = {}
    for sid in matched:
        st = data['structures'][sid]['state']
        by_state.setdefault(st, []).append(sid)

    print(f"Found {len(matched)} structures to reset:")
    for st, ids in sorted(by_state.items()):
        print(f"  {st}: {len(ids)}")
    print()

    if not args.clean:
        print("NOTE: Running without --clean flag")
        print("      Job states will be reset but Relax directories won't be removed")
        print("      Use --clean to remove Relax directories for fresh restart")
        print()

    # Create backup
    if not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(args.workflow).with_name(f"workflow_{timestamp}.json.bak")
        shutil.copy2(args.workflow, backup_path)
        print(f"Backup created: {backup_path}")

    # Process
    cleaned_count = 0
    for struct_id in sorted(matched):
        sdata = data['structures'][struct_id]
        old_state = sdata['state']
        print(f"  {struct_id}: {old_state} -> RELAX_FAILED")

        if not args.dry_run:
            sdata['state'] = 'RELAX_FAILED'
            sdata['relax_job_id'] = None
            sdata['error'] = f'Reset by simple_reset_to_relax.py (was {old_state})'

        if args.clean:
            relax_dir = Path(sdata.get('relax_dir', ''))
            if relax_dir.is_dir():
                if args.dry_run:
                    print(f"    [DRY RUN] Would remove: {relax_dir}")
                else:
                    shutil.rmtree(relax_dir)
                    print(f"    Removed: {relax_dir}")
                cleaned_count += 1

    # Save
    if not args.dry_run:
        with open(args.workflow, 'w') as f:
            json.dump(data, f, indent=2)

    # Summary
    print()
    print("=" * 67)
    print("Summary")
    print("=" * 67)
    print(f"  Structures reset to RELAX_FAILED: {len(matched)}")
    if args.clean:
        print(f"  Relax directories removed: {cleaned_count}")
    if args.dry_run:
        print("\n  DRY RUN - No changes were made")
    else:
        print("\n  Next step: run reset_failed_jobs.py to retry from scratch:")
        print("    python3 reset_failed_jobs.py --stage RELAX --clean")
    print("=" * 67)


if __name__ == '__main__':
    main()
