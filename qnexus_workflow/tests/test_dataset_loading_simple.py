#!/usr/bin/env python3
"""
Simple test script to verify dataset loading works correctly.
"""

import numpy as np
import json
import os

def test_load_problem():
    """Test loading a single problem from the dataset."""
    print("Testing dataset loading...")
    
    problems_dir = "problems"
    if not os.path.exists(problems_dir):
        print(f"Error: {problems_dir} directory not found!")
        return False
    
    # Test loading a 2x2 problem
    problem_name = "problem_2x2_0.5_10_3"
    
    try:
        A = np.load(os.path.join(problems_dir, f"{problem_name}_A.npy"))
        b = np.load(os.path.join(problems_dir, f"{problem_name}_b.npy"))
        csol = np.load(os.path.join(problems_dir, f"{problem_name}_csol.npy"))
        
        with open(os.path.join(problems_dir, f"{problem_name}_metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Successfully loaded {problem_name}")
        print(f"  Matrix A shape: {A.shape}")
        print(f"  Vector b shape: {b.shape}")
        print(f"  Solution shape: {csol.shape}")
        print(f"  Target sparsity: {metadata['sparsity']}")
        print(f"  Target condition number: {metadata['condition_number']}")
        
        # Verify solution
        residual = np.linalg.norm(A @ csol - b)
        print(f"  Solution residual: {residual:.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load {problem_name}: {e}")
        return False

def test_list_problems():
    """Test listing all available problems."""
    print("\nListing available problems...")
    
    problems_dir = "problems"
    problem_files = [f for f in os.listdir(problems_dir) if f.endswith('_A.npy')]
    problem_names = [f.replace('_A.npy', '') for f in problem_files]
    
    print(f"Found {len(problem_names)} problems:")
    
    # Group by size
    problems_by_size = {}
    for problem_name in problem_names:
        size = int(problem_name.split('x')[1].split('_')[0])
        if size not in problems_by_size:
            problems_by_size[size] = []
        problems_by_size[size].append(problem_name)
    
    for size in sorted(problems_by_size.keys()):
        print(f"  {size}×{size}: {len(problems_by_size[size])} problems")
        for problem_name in problems_by_size[size]:
            print(f"    - {problem_name}")
    
    return True

if __name__ == "__main__":
    print("Dataset Loading Test")
    print("=" * 40)
    
    success = test_load_problem()
    if success:
        test_list_problems()
        print("\n✓ Dataset loading test passed!")
    else:
        print("\n✗ Dataset loading test failed!")
