import itertools
from typing import List, Tuple, Optional, Dict
import numpy as np
import csv

def find_resistor_combinations(
    resistor_values_map: Dict[int, List[float]],
    target_min: float,
    target_max: float,
    resistor_constraints: Optional[Dict[int, Tuple[float, float]]] = None,
    allow_duplicates: bool = True,
    sort_results: bool = True,
    verbose: bool = True
) -> List[Tuple[float, float, float, float, float]]:
    """
    Find resistor combinations with separate value lists for each resistor.
    """
    
    if verbose:
        print(f"Target range: [{target_min}, {target_max}] Ohms")
        for idx in [1, 2, 3, 4]:
            if idx in resistor_values_map:
                print(f"  R{idx}: {len(resistor_values_map[idx])} available values")
    
    # Create separate lists for each position considering constraints
    available_by_pos = {}
    
    for pos in [1, 2, 3, 4]:
        if pos not in resistor_values_map:
            print(f"ERROR: No values provided for R{pos}!")
            return []
        
        # Start with the provided values for this resistor
        available = resistor_values_map[pos].copy()
        
        # Apply additional constraints if any
        if resistor_constraints and pos in resistor_constraints:
            min_val, max_val = resistor_constraints[pos]
            available = [r for r in available if min_val <= r <= max_val]
            if verbose:
                print(f"  Applied constraints R{pos} [{min_val}, {max_val}]: {len(available)} values remain")
        
        if not available:
            print(f"WARNING: No values available for R{pos} after constraints!")
            return []
        
        available_by_pos[pos] = sorted(available)
    
    # Total number of combinations
    total_combinations = (len(available_by_pos[1]) * 
                         len(available_by_pos[2]) * 
                         len(available_by_pos[3]) * 
                         len(available_by_pos[4]))
    
    if verbose:
        print(f"\nTotal combinations to check: {total_combinations:,}")
        print("-" * 70)
    
    def parallel(r_a, r_b):
        return (r_a * r_b) / (r_a + r_b) if (r_a + r_b) > 0 else float('inf')
    
    def total_resistance(r1, r2, r3, r4):
        return parallel(r1, r2) + parallel(r3, r4)
    
    solutions = []
    seen_combinations = set()
    
    # Generate combinations from available values for each position
    combinations = itertools.product(
        available_by_pos[1],
        available_by_pos[2],
        available_by_pos[3],
        available_by_pos[4]
    )
    
    checked = 0
    next_report = total_combinations / 10 if total_combinations > 10000 else total_combinations
    
    for r1, r2, r3, r4 in combinations:
        checked += 1
        
        if verbose and total_combinations > 10000 and checked >= next_report:
            print(f"Checked {checked:,}/{total_combinations:,} combinations... ({checked/total_combinations*100:.1f}%)")
            next_report += total_combinations / 10
        
        r_total = total_resistance(r1, r2, r3, r4)
        
        if target_min <= r_total <= target_max:
            # Avoid duplicates (R1,R2 can be swapped, and blocks can be swapped)
            pair1 = tuple(sorted([r1, r2]))
            pair2 = tuple(sorted([r3, r4]))
            canonical = (pair1[0], pair1[1], pair2[0], pair2[1])
            canonical_swapped = (pair2[0], pair2[1], pair1[0], pair1[1])
            
            if canonical not in seen_combinations and canonical_swapped not in seen_combinations:
                solutions.append((canonical[0], canonical[1], canonical[2], canonical[3], r_total))
                seen_combinations.add(canonical)
    
    if verbose:
        print(f"\nCheck complete. Solutions found: {len(solutions)}")
    
    if sort_results:
        solutions.sort(key=lambda x: x[4])
    
    return solutions

def calculate_accuracy_from_neighbors(sorted_values: List[float]) -> Dict[float, Tuple[float, float, float]]:
    """
    Calculate accuracy only for values that have BOTH neighbors.
    Accuracy = half the distance to the nearest neighbor.
    
    Returns: Dict[value, (left_neighbor, right_neighbor, accuracy_ohms)]
    Only includes values that have both left and right neighbors.
    """
    if len(sorted_values) < 3:  # Need at least 3 values to have one with both neighbors
        return {}
    
    accuracy_map = {}
    n = len(sorted_values)
    
    # Skip first and last elements (they don't have both neighbors)
    for i in range(1, n-1):
        val = sorted_values[i]
        left_neighbor = sorted_values[i-1]
        right_neighbor = sorted_values[i+1]
        
        # Calculate half-distances to neighbors
        left_half = (val - left_neighbor) / 2
        right_half = (right_neighbor - val) / 2
        
        # Accuracy is the minimum of the two half-distances
        accuracy = min(left_half, right_half)
        
        accuracy_map[val] = (left_neighbor, right_neighbor, accuracy)
    
    return accuracy_map

def print_worst_accuracy(solutions: List[Tuple]):
    """Find and print only the worst accuracy scenario for values with both neighbors"""
    if not solutions:
        print("No solutions found.")
        return
    
    # Get all total resistance values
    r_totals = [s[4] for s in solutions]
    
    if len(r_totals) < 3:
        print("Need at least 3 different resistance values to calculate accuracy (need values with both neighbors).")
        return
    
    # Calculate accuracy only for values with both neighbors
    accuracy_map = calculate_accuracy_from_neighbors(r_totals)
    
    if not accuracy_map:
        print("No values with both neighbors found.")
        return
    
    # Find worst accuracy (largest half-distance to nearest neighbor)
    worst_accuracy = max(data[2] for data in accuracy_map.values())
    
    # Find all values with this worst accuracy
    worst_values = [val for val, data in accuracy_map.items() if data[2] == worst_accuracy]
    
    print("\n" + "=" * 100)
    print("🔴 WORST ACCURACY SCENARIO (LARGEST GAP BETWEEN VALUES)")
    print("=" * 100)
    print(f"Worst accuracy: ±{worst_accuracy:.5f} Ω")
    
    for r_total in worst_values:
        left_neighbor, right_neighbor, accuracy = accuracy_map[r_total]
        
        print(f"\n📊 Value: {r_total:.5f} Ω")
        print(f"   Left neighbor: {left_neighbor:.5f} Ω")
        print(f"   Right neighbor: {right_neighbor:.5f} Ω")
        print(f"   Gap to left: {r_total - left_neighbor:.5f} Ω (half: {(r_total - left_neighbor)/2:.5f} Ω)")
        print(f"   Gap to right: {right_neighbor - r_total:.5f} Ω (half: {(right_neighbor - r_total)/2:.5f} Ω)")
        print(f"   Accuracy = min({(r_total - left_neighbor)/2:.5f}, {(right_neighbor - r_total)/2:.5f}) = {accuracy:.5f} Ω")
        print(f"   Coverage range: {r_total - accuracy:.5f} Ω to {r_total + accuracy:.5f} Ω")
        
        # Find all resistor combinations that give this total resistance
        matching_solutions = [s for s in solutions if abs(s[4] - r_total) < 1e-10]
        
        print(f"\n   Resistor combinations for {r_total:.5f} Ω (found {len(matching_solutions)}):")
        for i, (r1, r2, r3, r4, _) in enumerate(matching_solutions, 1):
            r12 = (r1 * r2) / (r1 + r2)
            r34 = (r3 * r4) / (r3 + r4)
            print(f"     #{i}: R1={r1:.5f}, R2={r2:.5f} → R12={r12:.5f} | R3={r3:.5f}, R4={r4:.5f} → R34={r34:.5f}")
    
    # Find the actual largest gap between any two consecutive values
    print("\n" + "=" * 100)
    
    # Find the largest gap between consecutive values
    max_gap = 0
    gap_left = None
    gap_right = None
    
    for i in range(len(r_totals) - 1):
        gap = r_totals[i+1] - r_totals[i]
        if gap > max_gap:
            max_gap = gap
            gap_left = r_totals[i]
            gap_right = r_totals[i+1]
    
    if gap_left is not None and gap_right is not None:
        print(f"Largest gap between consecutive values: {max_gap:.5f} Ω")
        print(f"between {gap_left:.5f} Ω and {gap_right:.5f} Ω")
        print(f"Each value's accuracy from this gap: ±{max_gap/2:.5f} Ω")
    
    print("=" * 100)

def save_to_csv(solutions: List[Tuple], target_range: Tuple[float, float], filename: str = "resistor_solutions.csv"):
    """
    Save all solutions to a CSV file with detailed information
    """
    target_min, target_max = target_range
    
    # Extract all total resistance values
    r_totals = [s[4] for s in solutions]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'Solution_Number', 
            'R1 (Ohm)', 
            'R2 (Ohm)', 
            'R12 (Ohm)', 
            'R3 (Ohm)', 
            'R4 (Ohm)', 
            'R34 (Ohm)', 
            'R_total (Ohm)'
        ])
        
        # Write each solution
        for i, (r1, r2, r3, r4, r_total) in enumerate(solutions, 1):
            r12 = (r1 * r2) / (r1 + r2)
            r34 = (r3 * r4) / (r3 + r4)
            
            writer.writerow([
                i,
                f"{r1:.5f}",
                f"{r2:.5f}",
                f"{r12:.5f}",
                f"{r3:.5f}",
                f"{r4:.5f}",
                f"{r34:.5f}",
                f"{r_total:.5f}"
            ])
    
    print(f"\n✅ Solutions saved to '{filename}'")

def filter_by_resistor_values(
    solutions: List[Tuple],
    resistor_idx: int,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> List[Tuple]:
    """Filter existing solutions by individual resistor constraints"""
    if not solutions:
        return []
    
    filtered = []
    idx_map = {1: 0, 2: 1, 3: 2, 4: 3}
    
    for sol in solutions:
        r_value = sol[idx_map[resistor_idx]]
        
        if min_val is not None and r_value < min_val:
            continue
        if max_val is not None and r_value > max_val:
            continue
        
        filtered.append(sol)
    
    return filtered

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    big_resistor = [
        100, 110, 120, 130, 150, 160, 180, 20, 22, 24, 27, 30,
        33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91
    ]

    small_resistor = [
        0.1, 1, 10, 1.1, 11, 1.2, 12, 1.3, 13, 1.5, 15, 1.6, 16,
        1.8, 18, 0.2, 2.0, 20, 2.2, 22, 2.4, 24, 2.7, 27, 0.3, 3.0,
        30, 3.3, 33, 3.6, 36, 3.9, 39, 4.3, 43, 4.7, 47, 0.51, 5.1,
        5.6, 6.2, 6.8, 0.75, 7.5, 8.2, 9.1
    ]

    resistor_values_map = {
        1: big_resistor,
        2: big_resistor,
        3: small_resistor,
        4: small_resistor
    }
    
    target_min = 50
    target_max = 500
    
    constraints = {
        1: (10, 100),
        2: (10, 100),
        3: (0, 20),
        4: (0, 20)
    }
    
    # Find all combinations
    solutions = find_resistor_combinations(
        resistor_values_map=resistor_values_map,
        target_min=target_min,
        target_max=target_max,
        resistor_constraints=constraints,
        allow_duplicates=True,
        verbose=True
    )
    
    # Print only worst accuracy scenario for values with both neighbors
    if solutions:
        print_worst_accuracy(solutions)
        
        # Save to CSV
        save_to_csv(solutions, target_range=(target_min, target_max), filename="resistor_solutions.csv")