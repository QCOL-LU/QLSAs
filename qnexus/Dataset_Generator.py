#!/usr/bin/env python3
"""
HHL Problem Dataset Generator
Creates a comprehensive dataset of linear system problems for HHL algorithm testing.

This script generates 20 problem instances:
- 5 instances each for matrix sizes: 2x2, 4x4, 8x8, and 16x16
- Varied sparsity levels (0.1 to 0.9)
- Varied condition numbers (2 to 50)
- Each problem includes matrix A, vector b, exact solution csol, and metadata
- Saves the dataset to the problems/ directory

Author: Generated for HHL Algorithm Testing
Date: 2024
"""

import numpy as np
import os
import json
import pandas as pd
from datetime import datetime
from Generate_Problem_V3 import generate_problem # use original, V2, or V3

def create_problem_instance(size, sparsity, cond_number, instance_num, seed=None):
    """
    Create a single problem instance with specified parameters.
    
    Parameters
    ----------
    size : int
        Matrix size (must be power of 2)
    sparsity : float
        Desired sparsity (0 to 1)
    cond_number : float
        Desired condition number
    instance_num : int
        Instance number for this parameter combination
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Problem instance with all data and metadata
    """
    if seed is None:
        seed = hash(f"{size}_{sparsity}_{cond_number}_{instance_num}") % (2**32)
    
    # Generate the problem
    problem = generate_problem(size, cond_number, sparsity, seed)
    
    # Create problem name
    problem_name = f"problem_{size}x{size}_{sparsity}_{cond_number}_{instance_num}"
    
    # Create instance data
    instance = {
        'problem_name': problem_name,
        'size': size,
        'sparsity': sparsity,
        'condition_number': cond_number,
        'instance_num': instance_num,
        'seed': seed,
        'matrix_A': problem['A'],
        'vector_b': problem['b'],
        'exact_solution': problem['csol'],
        'actual_condition_number': problem['condition_number'],
        'actual_sparsity': problem['sparsity'],
        'eigenvalues': problem['eigs'],
        'generation_timestamp': datetime.now().isoformat()
    }
    
    return instance

def generate_full_dataset():
    """
    Generate the complete HHL problem dataset.
    
    Returns
    -------
    list
        List of all problem instances
    """
    print("Generating HHL Problem Dataset...")
    print("=" * 50)
    
    # Define problem parameters
    sizes = [2, 4, 8, 16]
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    condition_numbers = [5, 10, 15, 20, 25]
    
    # Create dataset
    dataset = []
    
    for size in sizes:
        print(f"\nGenerating {size}x{size} problems...")
        for i in range(len(sparsities)):  # 5 instances per size
            # Use different parameter combinations for variety
            sparsity = sparsities[i % len(sparsities)]
            cond_num = condition_numbers[i % len(condition_numbers)]
            
            print(f"  Instance {i+1}: sparsity={sparsity}, cond_num={cond_num}")
            
            instance = create_problem_instance(size, sparsity, cond_num, i+1)
            dataset.append(instance)
    
    print(f"\nDataset generation complete! Created {len(dataset)} problem instances.")
    return dataset

def save_dataset_files(dataset, output_dir="problems"):
    """
    Save the dataset to individual files and create summary files.
    
    Parameters
    ----------
    dataset : list
        List of problem instances
    output_dir : str
        Directory to save the dataset files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving dataset files to '{output_dir}/' directory...")
    
    # Save individual problem files
    for instance in dataset:
        problem_name = instance['problem_name']
        
        # Save matrix A
        np.save(os.path.join(output_dir, f"{problem_name}_A.npy"), instance['matrix_A'])
        
        # Save vector b
        np.save(os.path.join(output_dir, f"{problem_name}_b.npy"), instance['vector_b'])
        
        # Save exact solution
        np.save(os.path.join(output_dir, f"{problem_name}_csol.npy"), instance['exact_solution'])
        
        # Save metadata (without numpy arrays)
        metadata = {k: v for k, v in instance.items() 
                   if not isinstance(v, np.ndarray)}
        with open(os.path.join(output_dir, f"{problem_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    # Create summary CSV
    summary_data = []
    for instance in dataset:
        summary_data.append({
            'problem_name': instance['problem_name'],
            'size': instance['size'],
            'target_sparsity': instance['sparsity'],
            'target_condition_number': instance['condition_number'],
            'actual_sparsity': instance['actual_sparsity'],
            'actual_condition_number': instance['actual_condition_number'],
            'instance_num': instance['instance_num'],
            'seed': instance['seed']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'dataset_summary.csv'), index=False)
    
    # Create complete metadata JSON
    complete_metadata = {
        'dataset_info': {
            'name': 'HHL Problem Dataset',
            'description': 'Linear system problems for HHL algorithm testing',
            'total_problems': len(dataset),
            'generation_date': datetime.now().isoformat(),
            'matrix_sizes': sorted(list(set(instance['size'] for instance in dataset))),
            'sparsity_range': [min(instance['sparsity'] for instance in dataset), 
                              max(instance['sparsity'] for instance in dataset)],
            'condition_number_range': [min(instance['condition_number'] for instance in dataset), 
                                     max(instance['condition_number'] for instance in dataset)]
        },
        'problems': [{k: v for k, v in instance.items() if not isinstance(v, np.ndarray)} 
                     for instance in dataset]
    }
    
    with open(os.path.join(output_dir, 'complete_dataset_metadata.json'), 'w') as f:
        json.dump(complete_metadata, f, indent=2, default=str)
    
    print(f"Dataset files saved successfully!")
    print(f"  - Individual problem files: {len(dataset) * 4}")
    print(f"  - Summary CSV: dataset_summary.csv")
    print(f"  - Complete metadata: complete_dataset_metadata.json")


def main():
    """Main function to generate and save the complete dataset."""
    print("HHL Problem Dataset Generator")
    print("=" * 50)
    
    # Generate the dataset
    dataset = generate_full_dataset()
    
    # Save all files
    save_dataset_files(dataset)
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 50)
    print(f"Total problems generated: {len(dataset)}")
    
    matrix_sizes = sorted(list(set(instance['size'] for instance in dataset)))
    print(f"Matrix sizes: {matrix_sizes}")
    
    min_sparsity = min(instance['sparsity'] for instance in dataset)
    max_sparsity = max(instance['sparsity'] for instance in dataset)
    print(f"Sparsity range: {min_sparsity:.1f} to {max_sparsity:.1f}")
    
    min_cond = min(instance['condition_number'] for instance in dataset)
    max_cond = max(instance['condition_number'] for instance in dataset)
    print(f"Condition number range: {min_cond} to {max_cond}")
    
    print(f"\nFiles saved in 'problems/' directory")
    print("Ready for distribution and use!")

if __name__ == "__main__":
    main() 
