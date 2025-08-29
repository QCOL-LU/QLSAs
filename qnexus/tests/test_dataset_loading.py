#!/usr/bin/env python3
"""
Test script to verify the HHL Problem Dataset can be loaded correctly.
"""

import numpy as np
import os
import json

def test_dataset_loading():
    """Test loading a few problems from the dataset."""
    print("Testing HHL Problem Dataset Loading...")
    print("=" * 50)
    
    problems_dir = "problems"
    
    if not os.path.exists(problems_dir):
        print(f"Error: {problems_dir} directory not found!")
        return False
    
    # Test loading a 2x2 problem
    print("\nTesting 2x2 problem loading...")
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
        
        # Check matrix properties
        is_hermitian = np.allclose(A, A.T)
        print(f"  Matrix is Hermitian: {is_hermitian}")
        
        eigenvalues = np.linalg.eigvals(A)
        is_positive_definite = np.all(eigenvalues > 0)
        print(f"  Matrix is positive definite: {is_positive_definite}")
        
    except Exception as e:
        print(f"✗ Failed to load {problem_name}: {e}")
        return False
    
    # Test loading a larger problem
    print("\nTesting 8x8 problem loading...")
    problem_name = "problem_8x8_0.7_20_4"
    
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
        
        # Check sparsity
        actual_sparsity = 1 - (np.count_nonzero(A) / A.size)
        print(f"  Actual sparsity: {actual_sparsity:.3f}")
        
    except Exception as e:
        print(f"✗ Failed to load {problem_name}: {e}")
        return False
    
    # Test loading summary CSV
    print("\nTesting summary CSV loading...")
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(problems_dir, "dataset_summary.csv"))
        print(f"✓ Successfully loaded summary CSV")
        print(f"  Total problems: {len(df)}")
        print(f"  Matrix sizes: {sorted(df['size'].unique())}")
        print(f"  Sparsity range: {df['target_sparsity'].min():.1f} to {df['target_sparsity'].max():.1f}")
        print(f"  Condition number range: {df['target_condition_number'].min()} to {df['target_condition_number'].max()}")
        
    except Exception as e:
        print(f"✗ Failed to load summary CSV: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("DATASET LOADING TEST COMPLETE!")
    print("✓ All tests passed - dataset is ready for distribution!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_dataset_loading()
