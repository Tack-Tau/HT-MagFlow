#!/usr/bin/env python3
"""
Search for binary rare-earth transition metal magnetic compositions.

Searches for transition metal-rich compositions with R and T elements.
"""

import json
from math import gcd
from typing import List, Dict, Tuple
from collections import defaultdict


# Element groups
RARE_EARTHS = ['Gd', 'Y']
TRANSITION_METALS = ['Fe', 'Co', 'Ni']


def search_binary_magnets(
    rare_earths: List[str] = RARE_EARTHS,
    transition_metals: List[str] = TRANSITION_METALS,
    max_atoms: int = 20,
    tm_rich_ratio: float = 0.75,
    max_compositions: int = -1
) -> List[Dict]:
    """
    Search for binary rare-earth transition metal magnetic compositions.
    
    Args:
        rare_earths: List of rare earth elements (R)
        transition_metals: List of transition metal elements (T)
        max_atoms: Maximum total atoms (N_R + N_T <= max_atoms)
        tm_rich_ratio: Minimum ratio for TM richness (N_T / (N_R + N_T) >= tm_rich_ratio)
        max_compositions: Maximum number of compositions to generate (-1 for all)
        
    Returns:
        List of valid compositions as dictionaries
    """
    valid_compositions = []
    n_count = 0
    
    print("Searching for binary rare-earth transition metal magnetic compositions...")
    print(f"Rare earths: {rare_earths}")
    print(f"Transition metals: {transition_metals}")
    print(f"TM-rich ratio: N_T / (N_R + N_T) >= {tm_rich_ratio}")
    print(f"Max atoms per composition: {max_atoms}")
    print("="*70)
    
    for R in rare_earths:
        for T in transition_metals:
            for n_r in range(1, max_atoms):
                for n_t in range(1, max_atoms):
                    if n_r + n_t > max_atoms:
                        continue
                    
                    g = gcd(n_r, n_t)
                    n_r_p, n_t_p = n_r // g, n_t // g
                    
                    if (n_r_p, n_t_p) != (n_r, n_t):
                        continue
                    
                    total = n_r_p + n_t_p
                    ratio = n_t_p / total
                    
                    if ratio >= tm_rich_ratio:
                        composition = {
                            R: n_r_p,
                            T: n_t_p
                        }
                        
                        formula = f"{R}{n_r_p}{T}{n_t_p}"
                        
                        valid_compositions.append({
                            'formula': formula,
                            'composition': composition,
                            'total_atoms': total,
                            'rare_earth': R,
                            'transition_metal': T,
                            'tm_ratio': ratio
                        })
                        
                        n_count += 1
                        
                        if n_count % 100 == 0:
                            print(f"Found {n_count} compositions... (Latest: {formula})")
                        
                        if max_compositions > 0 and n_count >= max_compositions:
                            print(f"\nReached maximum of {max_compositions} compositions.")
                            return valid_compositions
    
    print(f"\nTotal valid compositions found: {len(valid_compositions)}")
    return valid_compositions


def save_compositions(compositions: List[Dict], output_file: str = "binary_magnet_compositions.json"):
    """Save compositions to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(compositions, f, indent=2)
    print(f"Saved {len(compositions)} compositions to {output_file}")


def save_formulas_txt(compositions: List[Dict], output_file: str = "binary_magnet_compositions.txt"):
    """Save just formulas to text file for easy viewing."""
    with open(output_file, 'w') as f:
        for comp in compositions:
            f.write(f"{comp['formula']:15s} | TM ratio: {comp['tm_ratio']:.3f} | "
                   f"{comp['total_atoms']:2d} atoms\n")
    print(f"Saved formulas to {output_file}")


def print_statistics(compositions: List[Dict]):
    """Print composition statistics."""
    print("\n" + "="*80)
    print("BINARY MAGNET COMPOSITION SEARCH RESULTS")
    print("="*80)
    
    print(f"Total compositions found: {len(compositions)}")
    
    # Count by rare earth
    by_re = defaultdict(int)
    for comp in compositions:
        by_re[comp['rare_earth']] += 1
    
    # Count by transition metal
    by_tm = defaultdict(int)
    for comp in compositions:
        by_tm[comp['transition_metal']] += 1
    
    print(f"\nBy rare earth:")
    for re, count in sorted(by_re.items()):
        print(f"  {re}: {count} compositions")
    
    print(f"\nBy transition metal:")
    for tm, count in sorted(by_tm.items()):
        print(f"  {tm}: {count} compositions")
    
    # Show examples
    print("\nExample compositions:")
    for comp in compositions[:10]:
        print(f"  {comp['formula']:15s} | TM ratio: {comp['tm_ratio']:.3f} | "
              f"{comp['total_atoms']:2d} atoms")
    
    if len(compositions) > 10:
        print(f"  ... and {len(compositions) - 10} more")
    
    print("="*80)


def main():
    """Main execution."""
    print("="*80)
    print("BINARY RARE-EARTH TRANSITION METAL MAGNET SEARCH")
    print("="*80)
    
    # Search for compositions
    compositions = search_binary_magnets(
        rare_earths=RARE_EARTHS,
        transition_metals=TRANSITION_METALS,
        max_atoms=20,
        tm_rich_ratio=0.75,
        max_compositions=-1
    )
    
    # Print statistics
    print_statistics(compositions)
    
    # Save results
    save_compositions(compositions, "binary_magnet_compositions.json")
    save_formulas_txt(compositions, "binary_magnet_compositions.txt")
    
    print("\n" + "="*80)
    print(f" Search completed! Found {len(compositions)} binary magnet compositions.")
    print("="*80)


if __name__ == "__main__":
    main()

