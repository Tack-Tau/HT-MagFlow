#!/usr/bin/env python3
"""
Search for ternary rare-earth transition metal magnetic compositions.

Searches for transition metal-rich compositions with R, T, and T' elements.
"""

import json
from math import gcd
from functools import reduce
from typing import List, Dict, Tuple
from collections import defaultdict


# Element groups
RARE_EARTHS = ['Gd', 'Y']
TRANSITION_METALS_T = ['Fe', 'Co', 'Ni']
TRANSITION_METALS_TP = ['Ti', 'V', 'Cr', 'Mn', 'Cu', 'Zn']


def gcd_multiple(numbers: List[int]) -> int:
    """Calculate GCD of multiple numbers."""
    return reduce(gcd, numbers)


def search_ternary_magnets(
    rare_earths: List[str] = RARE_EARTHS,
    transition_metals_T: List[str] = TRANSITION_METALS_T,
    transition_metals_TP: List[str] = TRANSITION_METALS_TP,
    max_atoms: int = 20,
    tm_rich_ratio: float = 0.75,
    t_over_t_prime_ratio: float = 0.8,
    max_compositions: int = -1
) -> List[Dict]:
    """
    Search for ternary rare-earth transition metal magnetic compositions.
    
    Args:
        rare_earths: List of rare earth elements (R)
        transition_metals_T: List of primary transition metal elements (T)
        transition_metals_TP: List of secondary transition metal elements (T')
        max_atoms: Maximum total atoms (N_R + N_T + N_T' <= max_atoms)
        tm_rich_ratio: Minimum ratio for TM richness ((N_T + N_T') / (N_R + N_T + N_T') >= tm_rich_ratio)
        t_over_t_prime_ratio: Minimum ratio N_T / (N_T + N_T') (default: 0.8, only applied if T and T' lists differ)
        max_compositions: Maximum number of compositions to generate (-1 for all)
        
    Returns:
        List of valid compositions as dictionaries
    """
    valid_compositions = []
    n_count = 0
    
    same_lists = set(transition_metals_T) == set(transition_metals_TP)
    overlap = set(transition_metals_T) & set(transition_metals_TP)
    
    print("Searching for ternary rare-earth transition metal magnetic compositions...")
    print(f"Rare earths: {rare_earths}")
    print(f"Transition metals T: {transition_metals_T}")
    print(f"Transition metals T': {transition_metals_TP}")
    print(f"TM-rich ratio: (N_T + N_T') / (N_R + N_T + N_T') >= {tm_rich_ratio}")
    if not same_lists:
        print(f"T/T' ratio: N_T / (N_T + N_T') >= {t_over_t_prime_ratio}")
    if overlap:
        print(f"Overlap elements: {sorted(overlap)}")
    print(f"Max atoms per composition: {max_atoms}")
    print("="*70)
    
    for R in rare_earths:
        for T in transition_metals_T:
            for TP in transition_metals_TP:
                if T == TP:
                    continue
                # When both T and TP are in the overlap, enforce T < TP to avoid duplicates
                if T in overlap and TP in overlap and T > TP:
                    continue
                
                for n_r in range(1, max_atoms):
                    for n_t in range(1, max_atoms):
                        for n_tp in range(1, max_atoms):
                            if n_r + n_t + n_tp > max_atoms:
                                continue
                            
                            g = gcd_multiple([n_r, n_t, n_tp])
                            n_r_p, n_t_p, n_tp_p = n_r // g, n_t // g, n_tp // g
                            
                            if (n_r_p, n_t_p, n_tp_p) != (n_r, n_t, n_tp):
                                continue
                            
                            total = n_r_p + n_t_p + n_tp_p
                            
                            tm_ratio = (n_t_p + n_tp_p) / total
                            
                            if tm_ratio < tm_rich_ratio:
                                continue
                            
                            # Check T dominance ratio if lists differ
                            if not same_lists:
                                t_dom_ratio = n_t_p / (n_t_p + n_tp_p)
                                if t_dom_ratio < t_over_t_prime_ratio:
                                    continue
                            
                            composition = {
                                R: n_r_p,
                                T: n_t_p,
                                TP: n_tp_p
                            }
                            
                            formula = f"{R}{n_r_p}{T}{n_t_p}{TP}{n_tp_p}"
                            
                            valid_compositions.append({
                                'formula': formula,
                                'composition': composition,
                                'total_atoms': total,
                                'rare_earth': R,
                                'transition_metal_primary': T,
                                'transition_metal_secondary': TP,
                                'tm_ratio': tm_ratio
                            })
                            
                            n_count += 1
                            
                            if n_count % 100 == 0:
                                print(f"Found {n_count} compositions... (Latest: {formula})")
                            
                            if max_compositions > 0 and n_count >= max_compositions:
                                print(f"\nReached maximum of {max_compositions} compositions.")
                                return valid_compositions
    
    print(f"\nTotal valid compositions found: {len(valid_compositions)}")
    return valid_compositions


def save_compositions(compositions: List[Dict], output_file: str = "ternary_magnet_compositions.json"):
    """Save compositions to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(compositions, f, indent=2)
    print(f"Saved {len(compositions)} compositions to {output_file}")


def save_formulas_txt(compositions: List[Dict], output_file: str = "ternary_magnet_compositions.txt"):
    """Save just formulas to text file for easy viewing."""
    with open(output_file, 'w') as f:
        for comp in compositions:
            f.write(f"{comp['formula']:20s} | TM ratio: {comp['tm_ratio']:.3f} | "
                   f"{comp['total_atoms']:2d} atoms\n")
    print(f"Saved formulas to {output_file}")


def print_statistics(compositions: List[Dict]):
    """Print composition statistics."""
    print("\n" + "="*80)
    print("TERNARY MAGNET COMPOSITION SEARCH RESULTS")
    print("="*80)
    
    print(f"Total compositions found: {len(compositions)}")
    
    # Count by rare earth
    by_re = defaultdict(int)
    for comp in compositions:
        by_re[comp['rare_earth']] += 1
    
    # Count by transition metal primary
    by_tm_p = defaultdict(int)
    for comp in compositions:
        by_tm_p[comp['transition_metal_primary']] += 1
    
    # Count by transition metal secondary
    by_tm_s = defaultdict(int)
    for comp in compositions:
        by_tm_s[comp['transition_metal_secondary']] += 1
    
    print(f"\nBy rare earth:")
    for re, count in sorted(by_re.items()):
        print(f"  {re}: {count} compositions")
    
    print(f"\nBy transition metal T:")
    for tm, count in sorted(by_tm_p.items()):
        print(f"  {tm}: {count} compositions")
    
    print(f"\nBy transition metal T':")
    for tm, count in sorted(by_tm_s.items()):
        print(f"  {tm}: {count} compositions")
    
    # Show examples
    print("\nExample compositions:")
    for comp in compositions[:15]:
        print(f"  {comp['formula']:20s} | TM ratio: {comp['tm_ratio']:.3f} | "
              f"{comp['total_atoms']:2d} atoms")
    
    if len(compositions) > 15:
        print(f"  ... and {len(compositions) - 15} more")
    
    print("="*80)


def main():
    """Main execution."""
    print("="*80)
    print("TERNARY RARE-EARTH TRANSITION METAL MAGNET SEARCH")
    print("="*80)
    
    # Search for compositions
    compositions = search_ternary_magnets(
        rare_earths=RARE_EARTHS,
        transition_metals_T=TRANSITION_METALS_T,
        transition_metals_TP=TRANSITION_METALS_TP,
        max_atoms=20,
        tm_rich_ratio=0.75,
        t_over_t_prime_ratio=0.8,
        max_compositions=-1
    )
    
    # Print statistics
    print_statistics(compositions)
    
    # Save results
    save_compositions(compositions, "ternary_magnet_compositions.json")
    save_formulas_txt(compositions, "ternary_magnet_compositions.txt")
    
    print("\n" + "="*80)
    print(f" Search completed! Found {len(compositions)} ternary magnet compositions.")
    print("="*80)


if __name__ == "__main__":
    main()

