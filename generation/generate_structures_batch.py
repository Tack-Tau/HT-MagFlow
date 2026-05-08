#!/usr/bin/env python3
"""
Generate crystal structures for multiple compositions using MatterGen CSP mode.

This script supports both:
- Fine-tuned CSP checkpoints: Pass checkpoint directory path
- Pretrained models: Pass model name (e.g., "mattergen_base")

This script calls mattergen-generate for each composition via subprocess.
"""

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Dict
import argparse


def expand_composition_to_supercells(composition: Dict[str, int], max_atoms: int = 20) -> List[Dict[str, int]]:
    """
    Expand a reduced composition to all valid supercells up to max_atoms.
    
    For example, Li1B3P2 (6 atoms total) expands to:
    - 1x: Li1B3P2 (6 atoms)
    - 2x: Li2B6P4 (12 atoms)
    - 3x: Li3B9P6 (18 atoms)
    - 4x would be 24 atoms, exceeding max_atoms=20
    
    This allows MatterGen to explore different cell sizes in CSP mode.
    
    Args:
        composition: Reduced composition dict, e.g. {"Li": 1, "B": 3, "P": 2}
        max_atoms: Maximum total atoms allowed (mp_20 was trained on ≤20 atoms)
    
    Returns:
        List of composition dicts with different multipliers
    """
    # Calculate total atoms in reduced formula
    total_atoms_base = sum(composition.values())
    
    # Calculate how many multiples we can fit
    max_multiplier = max_atoms // total_atoms_base
    
    # Generate all valid supercells
    supercells = []
    for multiplier in range(1, max_multiplier + 1):
        supercell = {elem: count * multiplier for elem, count in composition.items()}
        total = sum(supercell.values())
        supercells.append(supercell)
    
    return supercells


def check_if_already_generated(output_dir: Path, min_structures: int = 1) -> bool:
    """
    Check if structures have already been generated for this composition.
    
    Args:
        output_dir: Output directory for composition
        min_structures: Minimum number of structures to consider complete
        
    Returns:
        True if already generated, False otherwise
    """
    if not output_dir.exists():
        return False
    
    # Check for combined output files
    extxyz_file = output_dir / "generated_crystals.extxyz"
    cif_zip = output_dir / "generated_crystals_cif.zip"
    
    # Consider complete if either file exists
    if extxyz_file.exists() or cif_zip.exists():
        return True
    
    return False


def generate_structures_for_composition(
    composition: Dict[str, int],
    formula: str,
    output_dir: Path,
    model_path: str,
    n_structures: int,
    max_atoms: int = 20,
    timeout: int = 1800,
    skip_if_exists: bool = True,
    structures_per_atom: float = None,
    max_batch_size: int = 100
) -> tuple[bool, str]:
    """
    Generate structures for a single composition using MatterGen CSP mode.
    
    This function expands the composition to all valid supercells (up to max_atoms)
    and generates equal numbers of structures for EACH supercell individually.
    Batch size is automatically calculated to minimize rounding waste.
    
    If structures_per_atom is provided, n_structures is recalculated proportional
    to the total atoms across all supercells.
    
    Args:
        composition: Dict like {"Li": 3, "Al": 1, "N": 2}
        formula: String formula like "Li3AlN2"
        output_dir: Output directory for structures
        model_path: Path to MatterGen model checkpoint or pretrained name
                   - Checkpoint: "outputs/singlerun/2025-10-10/15-30-45"
                   - Pretrained: "mattergen_base" or "mp_20_base"
        max_batch_size: Maximum structures per GPU batch (default: 100)
                       Larger requests are split into multiple batches to avoid OOM
        n_structures: Total number of structures to generate
        max_atoms: Maximum atoms per cell (default: 20 for mp_20)
        timeout: Timeout in seconds (default 30 min)
        skip_if_exists: Skip generation if structures already exist
        
    Returns:
        True if successful, False otherwise
    """
    # Check if already generated (for resume capability)
    if skip_if_exists and check_if_already_generated(output_dir, min_structures=1):
        print(f"    Already generated (skipping)")
        return True, "Already generated"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expand composition to all valid supercells
    supercells = expand_composition_to_supercells(composition, max_atoms)
    
    # Calculate total atoms across all supercells
    total_atoms_across_supercells = sum(sum(sc.values()) for sc in supercells)
    
    # If structures_per_atom specified, calculate n_structures proportionally
    if structures_per_atom is not None:
        n_structures = max(1, int(round(total_atoms_across_supercells * structures_per_atom)))
        print(f"  Proportional generation: {total_atoms_across_supercells} total atoms × {structures_per_atom} = {n_structures} structures")
    
    # Distribute structures across supercells proportional to atom count
    structures_per_supercell_list = []
    for sc in supercells:
        sc_atoms = sum(sc.values())
        # Proportional allocation: (atoms in this supercell / total atoms) * total structures
        n_struct = max(1, int(round(sc_atoms / total_atoms_across_supercells * n_structures)))
        structures_per_supercell_list.append(n_struct)
    
    # Recalculate actual total (due to rounding)
    total_structures_to_generate = sum(structures_per_supercell_list)
    
    print(f"  Configuration:")
    print(f"    Supercells: {len(supercells)}")
    print(f"    Target total structures: {n_structures}")
    print(f"    Distribution: Proportional to atom count per supercell")
    print(f"    Actual total structures: {total_structures_to_generate}")
    
    if total_structures_to_generate != n_structures:
        diff = total_structures_to_generate - n_structures
        print(f"    Note: {abs(diff)} structure{'s' if abs(diff) != 1 else ''} {'more' if diff > 0 else 'fewer'} due to rounding")
    
    # Print supercell info
    print(f"  Supercell sizes:")
    for i, sc in enumerate(supercells, 1):
        total = sum(sc.values())
        sc_formula = ''.join(f"{elem}{count}" for elem, count in sorted(sc.items()))
        n_struct = structures_per_supercell_list[i-1]
        print(f"    {i}x: {sc_formula} ({total} atoms) - {n_struct} structures")
    
    # Determine if model_path is a checkpoint directory or pretrained model name
    is_checkpoint = ('/' in model_path or Path(model_path).exists())
    
    # Generate structures for each supercell to temporary directories
    temp_dir = output_dir / ".temp_generation"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    all_success = True
    first_error = None
    total_generated = 0
    all_cif_files = []
    all_extxyz_entries = []
    
    for supercell_idx, supercell in enumerate(supercells, 1):
        sc_formula = ''.join(f"{elem}{count}" for elem, count in sorted(supercell.items()))
        sc_atoms = sum(supercell.values())
        
        # Get target structures for this specific supercell
        target_structures = structures_per_supercell_list[supercell_idx - 1]
        
        # Create temporary subdirectory for this supercell
        supercell_dir = temp_dir / f"supercell_{supercell_idx}x_{sc_atoms}atoms"
        supercell_dir.mkdir(parents=True, exist_ok=True)
        
        # Format composition as JSON list with single supercell
        comp_arg = json.dumps([supercell])
        
        # Split into multiple batches if target_structures exceeds max_batch_size
        # This prevents GPU OOM errors when generating many structures
        if target_structures <= max_batch_size:
            # Small enough - generate in one batch
            current_batch_size = target_structures
            num_batches = 1
            actual_structures = target_structures
        else:
            # Too large - split into multiple smaller batches
            current_batch_size = max_batch_size
            num_batches = (target_structures + max_batch_size - 1) // max_batch_size
            actual_structures = current_batch_size * num_batches
            
            if actual_structures != target_structures:
                print(f"        NOTE: Splitting {target_structures} structures into {num_batches} batches of {current_batch_size}")
                print(f"        Will generate {actual_structures} structures total (rounded up)")
        
        # Build mattergen-generate command
        cmd = [
            "mattergen-generate",
            str(supercell_dir),
        ]
        
        if is_checkpoint:
            cmd.append(f"--model_path={model_path}")
        else:
            cmd.append(f"--pretrained_name={model_path}")
        
        cmd.extend([
            "--sampling_config_name=csp",
            f"--target_compositions={comp_arg}",
            f"--batch_size={current_batch_size}",
            f"--num_batches={num_batches}",
            "--record_trajectories=False"
        ])
        
        if not is_checkpoint:
            cmd.extend([
                "--trainer.accelerator=gpu",
                "--trainer.devices=1",
                "--trainer.precision=32"
            ])
        
        if num_batches == 1:
            print(f"    [{supercell_idx}/{len(supercells)}] Generating {actual_structures} structures for {sc_formula} (batch_size={current_batch_size})")
            adjusted_timeout = timeout
        else:
            print(f"    [{supercell_idx}/{len(supercells)}] Generating {actual_structures} structures for {sc_formula} ({num_batches} batches × {current_batch_size} per batch)")
            adjusted_timeout = timeout * num_batches * 1.2
            print(f"        Timeout adjusted: {timeout}s → {int(adjusted_timeout)}s for {num_batches} batches")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=adjusted_timeout,
                check=False
            )
            
            # Collect generated CIF files (check both individual files and zip)
            cif_files = list(supercell_dir.glob("*.cif"))
            zip_file = supercell_dir / "generated_crystals_cif.zip"
            
            # If no individual CIF files but zip exists, extract and rename them
            if not cif_files and zip_file.exists():
                temp_cif_extract = supercell_dir / ".temp_cif_extract"
                temp_cif_extract.mkdir(exist_ok=True)
                
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    for name in zf.namelist():
                        if name.endswith('.cif'):
                            # Extract with renamed file (prefix with supercell name)
                            new_name = f"{supercell_dir.name}_{name}"
                            extract_path = temp_cif_extract / new_name
                            with open(extract_path, 'wb') as f:
                                f.write(zf.read(name))
                            cif_files.append(extract_path)
            
            # Check for extxyz file as alternative success indicator
            extxyz_file = supercell_dir / "generated_crystals.extxyz"
            extxyz_exists = extxyz_file.exists()
            
            # Success if either CIF files or extxyz file exists
            if cif_files or extxyz_exists:
                if cif_files:
                    all_cif_files.extend(cif_files)
                    n_generated = len(cif_files)
                    print(f"        Generated {n_generated} CIF structures for {sc_formula}")
                    total_generated += n_generated
                
                # Collect extxyz entries if file exists
                if extxyz_exists:
                    with open(extxyz_file, 'r') as f:
                        extxyz_content = f.read()
                        all_extxyz_entries.append(extxyz_content)
                        # Count structures in extxyz (count lines that start with a number followed by Properties=)
                        if not cif_files:
                            lines = extxyz_content.split('\n')
                            n_structures = 0
                            for i, line in enumerate(lines):
                                if line.strip() and line.strip()[0].isdigit():
                                    # Check if next line contains Lattice= (extxyz format)
                                    if i+1 < len(lines) and 'Lattice=' in lines[i+1]:
                                        n_structures += 1
                            print(f"        Generated {n_structures} structures for {sc_formula} (extxyz only)")
                            total_generated += n_structures
            else:
                # Debug: List all files created in the directory
                all_files = list(supercell_dir.glob("*"))
                print(f"        DEBUG: No CIF or extxyz files found. All files in {supercell_dir.name}:")
                for f in all_files[:20]:  # Show first 20 files
                    print(f"          - {f.name} ({f.stat().st_size} bytes)")
                if len(all_files) > 20:
                    print(f"          ... and {len(all_files)-20} more files")
                if not all_files:
                    print(f"          (directory is empty)")
                error_msg = f"Exit code: {result.returncode}"
                if first_error is None:
                    first_error = f"{formula} (supercell: {sc_formula}): {error_msg}"
                    if result.stderr:
                        first_error += f" | stderr: {result.stderr[:200]}"
                
                print(f"        ERROR: Failed to generate structures for supercell {sc_formula} of {formula}")
                print(f"        Exit code: {result.returncode}")
                print(f"        Command: {' '.join(cmd)}")
                if result.stderr:
                    print(f"        STDERR (first 500 chars):")
                    print("        " + "\n        ".join(result.stderr[:500].split('\n')))
                if result.stdout:
                    print(f"        STDOUT (last 300 chars):")
                    print("        " + "\n        ".join(result.stdout[-300:].split('\n')))
                all_success = False
                
        except subprocess.TimeoutExpired:
            if first_error is None:
                first_error = f"{formula} (supercell: {sc_formula}): Timeout after {timeout} seconds"
            print(f"        ERROR: Timeout for supercell {sc_formula} of {formula} after {timeout} seconds")
            all_success = False
        except Exception as e:
            if first_error is None:
                first_error = f"{formula} (supercell: {sc_formula}): Exception {type(e).__name__}: {str(e)[:100]}"
            print(f"        ERROR: Exception for supercell {sc_formula} of {formula}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_success = False
    
    # Combine all generated files
    if all_cif_files or all_extxyz_entries:
        if all_cif_files:
            print(f"  Combining {len(all_cif_files)} CIF structures from {len(supercells)} supercells...")
            
            # Create combined zip file
            combined_zip = output_dir / "generated_crystals_cif.zip"
            with zipfile.ZipFile(combined_zip, 'w') as zf:
                for cif_file in all_cif_files:
                    zf.write(cif_file, arcname=cif_file.name)
        
        # Create combined extxyz file
        if all_extxyz_entries:
            if not all_cif_files:
                print(f"  Combining {total_generated} structures from {len(supercells)} supercells (extxyz only)...")
            combined_extxyz = output_dir / "generated_crystals.extxyz"
            with open(combined_extxyz, 'w') as f:
                # Concatenate extxyz files directly (no blank lines between)
                # Each entry is already a complete extxyz file from one supercell
                for content in all_extxyz_entries:
                    # Ensure content ends with exactly one newline
                    if not content.endswith('\n'):
                        content += '\n'
                    f.write(content)
        
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"  Success! Generated {total_generated} total structures across {len(supercells)} supercells")
        if all_cif_files and all_extxyz_entries:
            print(f"  Output: {combined_zip.name}, {combined_extxyz.name}")
        elif all_cif_files:
            print(f"  Output: {combined_zip.name} (no extxyz)")
        else:
            print(f"  Output: {combined_extxyz.name} (extxyz only)")
        return True, "Success"
    else:
        # Keep temp_dir for debugging when generation fails
        error_detail = first_error if first_error else "No structures generated"
        print(f"  Failed: {error_detail}")
        print(f"  NOTE: Temporary directory preserved for debugging: {temp_dir}")
        return False, error_detail


def process_compositions(
    compositions_file: Path,
    output_base_dir: Path,
    model_path: str,
    n_structures: int,
    max_atoms: int = 20,
    max_compositions: int = -1,
    start_index: int = 0,
    skip_existing: bool = True,
    structures_per_atom: float = None,
    max_batch_size: int = 100,
    timeout: int = 1800
) -> Dict:
    """
    Process multiple compositions and generate structures.
    
    Args:
        compositions_file: JSON file with composition data
        output_base_dir: Base directory for all outputs
        model_path: MatterGen model path
        n_structures: Structures to generate per composition
        max_atoms: Maximum atoms per cell (for supercell expansion)
        max_compositions: Max compositions to process (-1 for all)
        start_index: Starting index in compositions list
        skip_existing: Skip compositions that already have generated structures
        structures_per_atom: Structures per atom (proportional mode)
        max_batch_size: Maximum structures per GPU batch
        timeout: Base timeout in seconds per batch
    Returns:
        Dictionary with generation statistics
    """
    # Load compositions
    with open(compositions_file) as f:
        compositions = json.load(f)
    
    # Apply slicing
    end_index = start_index + max_compositions if max_compositions > 0 else len(compositions)
    compositions = compositions[start_index:end_index]
    
    # Determine model type
    is_checkpoint = ('/' in model_path or Path(model_path).exists())
    model_type = "Fine-tuned checkpoint" if is_checkpoint else "Pretrained model"
    
    print("="*70)
    print(f"BATCH STRUCTURE GENERATION (CSP MODE)")
    print("="*70)
    print(f"Compositions file: {compositions_file}")
    print(f"Processing: {len(compositions)} compositions")
    print(f"Range: {start_index} to {end_index}")
    print(f"Model: {model_path}")
    print(f"Model type: {model_type}")
    if structures_per_atom is not None:
        print(f"Generation mode: Proportional ({structures_per_atom} structures per atom)")
    else:
        print(f"Generation mode: Fixed ({n_structures} structures per composition)")
    print(f"Max atoms per cell: {max_atoms} (supercell expansion enabled)")
    print(f"Batch size: Exact count per supercell (proportional to atoms)")
    print(f"Resume mode: {'Enabled (skip existing)' if skip_existing else 'Disabled (regenerate all)'}")
    print("="*70)
    
    success_count = 0
    skipped_count = 0
    failed_compositions = []
    failed_details = {}  # Store detailed error info
    
    for idx, comp_data in enumerate(compositions, start=1):
        formula = comp_data['formula']
        composition = comp_data['composition']
        
        print(f"\n[{idx}/{len(compositions)}] Generating: {formula}")
        print(f"  Composition: {composition}")
        if 'excess_electrons' in comp_data:
            print(f"  Excess electrons: {comp_data['excess_electrons']}")
        print(f"  Total atoms: {comp_data['total_atoms']}")
        
        # Output directory for this composition
        output_dir = output_base_dir / f"{formula}_structures"
        
        # Check if already exists before attempting generation
        if skip_existing and check_if_already_generated(output_dir, min_structures=1):
            print(f"    Already generated (skipping)")
            skipped_count += 1
            success_count += 1
            continue
        
        # Generate structures
        success, error_msg = generate_structures_for_composition(
            composition=composition,
            formula=formula,
            output_dir=output_dir,
            model_path=model_path,
            n_structures=n_structures,
            max_atoms=max_atoms,
            skip_if_exists=False,  # Already checked above
            structures_per_atom=structures_per_atom,
            max_batch_size=max_batch_size,
            timeout=timeout
        )
        
        if success:
            success_count += 1
        else:
            failed_compositions.append(formula)
            failed_details[formula] = error_msg
        
        # Progress update every 10 compositions
        if idx % 10 == 0:
            print(f"\n--- Progress: {idx}/{len(compositions)} ({success_count} successful, {skipped_count} skipped) ---")
    
    # Generate statistics
    stats = {
        "total_processed": len(compositions),
        "successful": success_count,
        "skipped": skipped_count,
        "newly_generated": success_count - skipped_count,
        "failed": len(failed_compositions),
        "success_rate": success_count / len(compositions) * 100 if compositions else 0,
        "failed_compositions": failed_compositions
    }
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Total compositions processed: {stats['total_processed']}")
    print(f"Successful: {stats['successful']}")
    if skipped_count > 0:
        print(f"  - Skipped (already existed): {skipped_count}")
        print(f"  - Newly generated: {stats['newly_generated']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    
    if failed_compositions:
        print("\nFailed compositions (with error details):")
        for formula in failed_compositions[:10]:
            error_detail = failed_details.get(formula, "Unknown error")
            print(f"  - {formula}: {error_detail}")
        if len(failed_compositions) > 10:
            print(f"  ... and {len(failed_compositions)-10} more")
        
        # Save failed compositions
        failed_file = output_base_dir / "failed_compositions.txt"
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed_compositions))
        print(f"\nFailed compositions saved to: {failed_file}")
        
        # Save detailed error log
        error_log_file = output_base_dir / "failed_compositions_detailed.txt"
        with open(error_log_file, 'w') as f:
            for formula in failed_compositions:
                error_detail = failed_details.get(formula, "Unknown error")
                f.write(f"{formula}: {error_detail}\n")
        print(f"Detailed error log saved to: {error_log_file}")
    
    print("="*70)
    
    return stats


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate crystal structures for multiple compositions using MatterGen"
    )
    parser.add_argument(
        "--compositions", "-c",
        type=str,
        required=True,
        help="JSON file with compositions"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Base output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mattergen_base",
        help="MatterGen model: checkpoint path (e.g., outputs/singlerun/2025-10-10/15-30-45) or pretrained name (e.g., mattergen_base)"
    )
    parser.add_argument(
        "--n-structures", "-n",
        type=int,
        default=20,
        help="Number of structures per composition (default: 20). Ignored if --structures-per-atom is set."
    )
    parser.add_argument(
        "--structures-per-atom",
        type=float,
        default=None,
        help="Structures per atom (proportional mode). If set, overrides --n-structures. E.g., 2.0 generates 2 structures per atom across all supercells."
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=20,
        help="Maximum atoms per cell for supercell expansion (default: 20, matching mp_20 dataset)"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=100,
        help="Maximum structures per GPU batch (default: 100). Larger requests split into multiple batches to avoid OOM."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Base timeout in seconds per batch (default: 1800 = 30 min). Auto-scaled for multiple batches."
    )
    parser.add_argument(
        "--max-compositions", "-m",
        type=int,
        default=-1,
        help="Maximum compositions to process, -1 for all (default: -1)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index in compositions list (default: 0)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip compositions that already have generated structures (default: True, enables resume)"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Regenerate all structures, even if they already exist"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    compositions_file = Path(args.compositions)
    output_base_dir = Path(args.output_dir)
    
    # Validate inputs
    if not compositions_file.exists():
        print(f"ERROR: Compositions file not found: {compositions_file}")
        sys.exit(1)
    
    # Run generation
    stats = process_compositions(
        compositions_file=compositions_file,
        output_base_dir=output_base_dir,
        model_path=args.model,
        n_structures=args.n_structures,
        max_atoms=args.max_atoms,
        max_compositions=args.max_compositions,
        start_index=args.start_index,
        skip_existing=args.skip_existing,
        structures_per_atom=args.structures_per_atom,
        max_batch_size=args.max_batch_size,
        timeout=args.timeout
    )
    
    # Save statistics
    stats_file = output_base_dir / "generation_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
