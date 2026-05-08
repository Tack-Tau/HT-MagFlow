#!/usr/bin/env python3
"""
Generate derivative structures from Ce-Fe prototype CIFs
Substitutes Ce with rare earth elements and Fe with 3d transition metals.

Output format compatible with VASPflow prescreen workflow:
  - Creates *_structures/ directories
  - Each contains generated_crystals_cif.zip with CIF files
"""

import os
import sys
import zipfile
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter

# Element substitution lists
A_ELEMENTS = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
B_ELEMENTS = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']


def substitute_elements(structure, substitutions):
    """
    Substitute elements in a structure.
    
    Args:
        structure: pymatgen Structure object
        substitutions: dict mapping old_element -> new_element (e.g., {'Ce': 'Pr', 'Fe': 'Ti'})
    
    Returns:
        New Structure with substituted elements
    """
    new_species = []
    for site in structure:
        old_el = str(site.specie)
        new_el = substitutions.get(old_el, old_el)
        new_species.append(new_el)
    
    new_structure = Structure(
        lattice=structure.lattice,
        species=new_species,
        coords=structure.frac_coords,
        coords_are_cartesian=False
    )
    
    return new_structure


def get_composition_formula(structure):
    """Get reduced composition formula (e.g., Pr10Ti52 -> Pr5Ti26)."""
    comp = structure.composition
    return comp.reduced_formula


def generate_derivatives(prototype_path, output_dir, a_elements, b_elements, 
                        a_symbol='Ce', b_symbol='Fe', max_per_proto=None):
    """
    Generate derivative structures from a prototype.
    
    Args:
        prototype_path: Path to prototype CIF file
        output_dir: Output directory for results
        a_elements: List of elements to substitute for A-type (Ce)
        b_elements: List of elements to substitute for B-type (Fe)
        a_symbol: Symbol to replace with A-type elements
        b_symbol: Symbol to replace with B-type elements
        max_per_proto: Maximum derivatives per prototype (None = all)
    """
    prototype_path = Path(prototype_path)
    output_dir = Path(output_dir)
    
    print(f"\nProcessing prototype: {prototype_path.name}")
    
    # Parse prototype structure
    parser = CifParser(str(prototype_path))
    proto_structure = parser.parse_structures(primitive=False)[0]
    
    print(f"  Original composition: {proto_structure.composition.formula}")
    
    # Check which elements need substitution
    elements_in_structure = [str(el) for el in proto_structure.composition.elements]
    has_a = a_symbol in elements_in_structure
    has_b = b_symbol in elements_in_structure
    
    if not has_a and not has_b:
        print(f"  Warning: Neither {a_symbol} nor {b_symbol} found in structure")
        return 0
    
    generated_count = 0
    
    # Generate all A-B combinations
    for a_el in a_elements:
        for b_el in b_elements:
            if max_per_proto and generated_count >= max_per_proto:
                break
            
            # Create substitution map
            substitutions = {}
            if has_a:
                substitutions[a_symbol] = a_el
            if has_b:
                substitutions[b_symbol] = b_el
            
            # Generate new structure
            new_structure = substitute_elements(proto_structure, substitutions)
            comp_formula = get_composition_formula(new_structure)
            
            # Create output directory structure
            comp_dir = output_dir / f"{comp_formula}_structures"
            comp_dir.mkdir(parents=True, exist_ok=True)
            
            # CIF filename inside zip
            cif_filename = f"{comp_formula}_s001.cif"
            
            # Create zip file with single CIF
            zip_path = comp_dir / "generated_crystals_cif.zip"
            
            # Write CIF to zip
            writer = CifWriter(new_structure, symprec=0.01)
            cif_string = str(writer)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(cif_filename, cif_string)
            
            generated_count += 1
    
    print(f"  Generated {generated_count} derivative structures")
    return generated_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Ce-Fe derivative structures for VASPflow screening"
    )
    parser.add_argument(
        '--prototype-dir',
        type=str,
        default='./',
        help="Directory containing prototype CIF files"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./CeFe_derivatives_results',
        help="Output directory for generated structures"
    )
    parser.add_argument(
        '--prototypes',
        type=str,
        nargs='+',
        default=['Ce10Fe52.cif', 'Ce2Fe14B.cif'],
        help="Prototype CIF filenames"
    )
    parser.add_argument(
        '--max-per-proto',
        type=int,
        default=None,
        help="Maximum derivatives per prototype (default: all combinations)"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help="Test mode: generate only 5 structures per prototype"
    )
    
    args = parser.parse_args()
    
    prototype_dir = Path(args.prototype_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    
    if args.test:
        args.max_per_proto = 5
    
    print("="*70)
    print("Ce-Fe Derivative Structure Generator")
    print("="*70)
    print(f"Prototype directory: {prototype_dir}")
    print(f"Output directory: {output_dir}")
    print(f"A-type elements (Ce substitution): {len(A_ELEMENTS)} - {', '.join(A_ELEMENTS)}")
    print(f"B-type elements (Fe substitution): {len(B_ELEMENTS)} - {', '.join(B_ELEMENTS)}")
    print(f"Total combinations per prototype: {len(A_ELEMENTS) * len(B_ELEMENTS)} = {len(A_ELEMENTS)} × {len(B_ELEMENTS)}")
    if args.max_per_proto:
        print(f"Limiting to: {args.max_per_proto} per prototype")
    print("="*70)
    
    # Check prototype files exist
    for proto_file in args.prototypes:
        proto_path = prototype_dir / proto_file
        if not proto_path.exists():
            print(f"Error: Prototype file not found: {proto_path}")
            return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each prototype
    total_generated = 0
    
    for proto_file in args.prototypes:
        proto_path = prototype_dir / proto_file
        
        count = generate_derivatives(
            prototype_path=proto_path,
            output_dir=output_dir,
            a_elements=A_ELEMENTS,
            b_elements=B_ELEMENTS,
            a_symbol='Ce',
            b_symbol='Fe',
            max_per_proto=args.max_per_proto
        )
        
        total_generated += count
    
    print("\n" + "="*70)
    print("Generation Complete")
    print("="*70)
    print(f"Total structures generated: {total_generated}")
    print(f"Output directory: {output_dir}")
    print("")
    print("Directory structure:")
    comp_dirs = sorted(output_dir.glob("*_structures"))
    if comp_dirs:
        print(f"  {len(comp_dirs)} composition directories created")
        print(f"  Example: {comp_dirs[0].name}")
        for comp_dir in comp_dirs[:3]:
            zip_file = comp_dir / "generated_crystals_cif.zip"
            if zip_file.exists():
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    print(f"    {comp_dir.name}/generated_crystals_cif.zip → {zf.namelist()[0]}")
        if len(comp_dirs) > 3:
            print(f"    ... and {len(comp_dirs) - 3} more")
    print("")
    print("Next steps:")
    print(f"  1. Transfer to HPC: scp -r {output_dir} your_cluster:/scratch/$USER/")
    print(f"  2. Run pre-screening:")
    print(f"       bash run_prescreen.sh \\")
    print(f"         --results-dir {output_dir.name} \\")
    print(f"         --output-dir ./VASP_JOBS \\")
    print(f"         --device cuda")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

