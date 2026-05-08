#!/usr/bin/env python3
"""
Create summary of generated structures across all compositions.
Supports both individual CIF files and generated_crystals_cif.zip archives.
"""

import json
import argparse
import zipfile
from pathlib import Path
from typing import Dict, List


def summarize_generation(output_base_dir: Path) -> Dict:
    """
    Create summary of all generated structures.
    
    Args:
        output_base_dir: Base directory containing all structure subdirectories
        
    Returns:
        Dictionary with summary statistics
    """
    print("="*70)
    print("CREATING GENERATION SUMMARY")
    print("="*70)
    print(f"Scanning directory: {output_base_dir}")
    
    # Find all structure directories
    structure_dirs = sorted(output_base_dir.glob("*_structures"))
    
    summary = {
        "total_compositions": len(structure_dirs),
        "compositions": []
    }
    
    total_structures = 0
    
    for struct_dir in structure_dirs:
        formula = struct_dir.name.replace("_structures", "")
        
        n_structures = 0
        
        # Count structures: check ZIP file first, then extxyz, then individual CIF files
        cif_zip = struct_dir / "generated_crystals_cif.zip"
        extxyz_file = struct_dir / "generated_crystals.extxyz"
        
        if cif_zip.exists():
            # Count files in ZIP
            try:
                with zipfile.ZipFile(cif_zip, 'r') as zf:
                    cif_files_in_zip = [f for f in zf.namelist() if f.endswith('.cif')]
                    n_structures = len(cif_files_in_zip)
            except Exception as e:
                print(f"  Warning: Could not read {cif_zip}: {e}")
                n_structures = 0
        
        # If no CIFs found in zip, try counting from extxyz
        if n_structures == 0 and extxyz_file.exists():
            try:
                with open(extxyz_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    # Count structures in extxyz (each structure starts with atom count line followed by Lattice= line)
                    for i, line in enumerate(lines):
                        if line.strip() and line.strip()[0].isdigit():
                            # Check if next line contains Lattice= (extxyz format)
                            if i+1 < len(lines) and 'Lattice=' in lines[i+1]:
                                n_structures += 1
            except Exception as e:
                print(f"  Warning: Could not read {extxyz_file}: {e}")
        
        # If still no structures, count individual CIF files
        if n_structures == 0:
            cif_files = list(struct_dir.glob("*.cif"))
            n_structures = len(cif_files)
        
        total_structures += n_structures
        
        # Load metadata if exists
        metadata_file = struct_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except:
                pass
        
        # Check for properties file
        properties_file = struct_dir / "properties.csv"
        has_properties = properties_file.exists()
        
        comp_info = {
            "formula": formula,
            "structures_generated": n_structures,
            "output_directory": str(struct_dir),
            "has_metadata": len(metadata) > 0,
            "has_properties": has_properties,
        }
        
        # Add some metadata if available
        if metadata:
            comp_info["generation_time"] = metadata.get("generation_time")
            comp_info["model_used"] = metadata.get("model_path")
        
        summary["compositions"].append(comp_info)
        
        print(f"  {formula:20s} : {n_structures:3d} structures")
    
    # Calculate statistics
    summary["total_structures_generated"] = total_structures
    summary["average_structures_per_composition"] = (
        total_structures / len(structure_dirs) if structure_dirs else 0
    )
    
    # Distribution statistics
    structure_counts = [c["structures_generated"] for c in summary["compositions"]]
    if structure_counts:
        summary["min_structures"] = min(structure_counts)
        summary["max_structures"] = max(structure_counts)
        summary["median_structures"] = sorted(structure_counts)[len(structure_counts)//2]
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total compositions: {summary['total_compositions']}")
    print(f"Total structures generated: {summary['total_structures_generated']}")
    print(f"Average structures per composition: {summary['average_structures_per_composition']:.1f}")
    if structure_counts:
        print(f"Min structures: {summary['min_structures']}")
        print(f"Max structures: {summary['max_structures']}")
        print(f"Median structures: {summary['median_structures']}")
    print("="*70)
    
    return summary


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Summarize generated structures"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Base output directory with generated structures"
    )
    parser.add_argument(
        "--summary-file", "-s",
        type=str,
        default=None,
        help="Output summary JSON file (default: <output-dir>/generation_summary.json)"
    )
    
    args = parser.parse_args()
    
    output_base_dir = Path(args.output_dir)
    
    if not output_base_dir.exists():
        print(f"ERROR: Output directory not found: {output_base_dir}")
        return 1
    
    # Generate summary
    summary = summarize_generation(output_base_dir)
    
    # Save summary
    if args.summary_file:
        summary_file = Path(args.summary_file)
    else:
        summary_file = output_base_dir / "generation_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

