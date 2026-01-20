# HHL Problem Dataset

This dataset contains 20 linear system problems specifically designed for testing and benchmarking HHL (Harrow-Hassidim-Lloyd) quantum algorithms.

## Dataset Overview

- **Total Problems**: 20
- **Matrix Sizes**: 2x2, 4x4, 8x8, 16x16 (5 instances each)
- **Sparsity Levels**: 0.1, 0.3, 0.5, 0.7, 0.9
- **Condition Numbers**: 5, 10, 15, 20, 25

## File Structure

Each problem instance consists of 4 files:

```
problem_{size}x{size}_{sparsity}_{condition_number}_{instance}/
├── {problem_name}_A.npy          # Matrix A (numpy array)
├── {problem_name}_b.npy          # Vector b (numpy array)
├── {problem_name}_csol.npy       # Exact solution (numpy array)
└── {problem_name}_metadata.json  # Problem metadata
```

## File Descriptions

### Matrix A (.npy)
- **Format**: numpy array
- **Shape**: (n, n) where n is the matrix size
- **Properties**: Hermitian, positive definite
- **Content**: Coefficient matrix of the linear system Ax = b

### Vector b (.npy)
- **Format**: numpy array
- **Shape**: (n,) where n is the matrix size
- **Content**: Right-hand side vector of the linear system Ax = b

### Exact Solution (.npy)
- **Format**: numpy array
- **Shape**: (n,) where n is the matrix size
- **Content**: Classical solution x = A^(-1) * b

### Metadata (.json)
- **Format**: JSON file
- **Content**: Problem parameters, actual achieved values, generation info

## Usage Examples

### Python
```python
import numpy as np

# Load a problem
problem_name = "problem_4x4_0.5_10_1"
A = np.load(f"{problem_name}_A.npy")
b = np.load(f"{problem_name}_b.npy")
csol = np.load(f"{problem_name}_csol.npy")

# Verify the solution
residual = np.linalg.norm(A @ csol - b)
print(f"Residual: {residual}")
```

### MATLAB
```matlab
% Load a problem
problem_name = 'problem_4x4_0.5_10_1';
A = load([problem_name '_A.npy']);
b = load([problem_name '_b.npy']);
csol = load([problem_name '_csol.npy']);

% Verify the solution
residual = norm(A * csol - b);
fprintf('Residual: %e\n', residual);
```

## Problem Parameters

### Sparsity
- **0.1**: Very dense (90% non-zero elements)
- **0.3**: Moderately dense (70% non-zero elements)
- **0.5**: Balanced (50% non-zero elements)
- **0.7**: Moderately sparse (30% non-zero elements)
- **0.9**: Very sparse (10% non-zero elements)

### Condition Number
- **2**: Well-conditioned (easy to solve)
- **5**: Moderately conditioned
- **10**: Moderately ill-conditioned
- **20**: Ill-conditioned (challenging)
- **50**: Very ill-conditioned (difficult)

## Matrix Properties

All matrices in this dataset:
- Are **Hermitian** (A = A^†)
- Are **positive definite** (all eigenvalues > 0)
- Have **power-of-2 dimensions** (required for HHL algorithm)
- Are **normalized** for numerical stability
- Have **controlled sparsity** patterns

## Reproducibility

All problems are generated with fixed seeds, ensuring:
- Identical results across different runs
- Reproducible benchmarks
- Consistent testing conditions

## Citation

If you use this dataset in your research, please cite:
```
HHL Problem Dataset for Quantum Algorithm Testing
Generated using Generate_Problem_V2.py
```

## Contact

For questions about this dataset, please refer to the original code repository.
