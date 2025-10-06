# HHL Problem Dataset - Distribution Summary

## 🎯 Dataset Overview

**Total Problems**: 20 linear system instances  
**Purpose**: Testing and benchmarking HHL (Harrow-Hassidim-Lloyd) quantum algorithms  
**Matrix Sizes**: 2x2, 4x4, 8x8, 16x16 (5 instances each)  
**Parameter Range**: Sparsity, condition number

## 📁 What's Included

The `problems/` directory contains:

### **Individual Problem Files (80 total)**
- `{problem_name}_A.npy` - Matrix A (numpy array)
- `{problem_name}_b.npy` - Vector b (numpy array)  
- `{problem_name}_csol.npy` - Exact solution (numpy array)
- `{problem_name}_metadata.json` - Problem metadata

### **Summary Files**
- `dataset_summary.csv` - Overview table of all problems
- `complete_dataset_metadata.json` - Complete metadata in JSON format
- `README.md` - Comprehensive usage documentation

## 🚀 How to Distribute

### **Option 1: Zip the entire `problems/` directory**
```bash
zip -r HHL_Problem_Dataset.zip problems/
```

### **Option 2: Create a tar archive**
```bash
tar -czf HHL_Problem_Dataset.tar.gz problems/
```

### **Option 3: Share individual files**
- Send the entire `problems/` folder
- Include the main `README.md` file
- Recipients can use any of the summary files for reference

## 📋 Problem Naming Convention

```
problem_{size}x{size}_{sparsity}_{condition_number}_{instance}
```

**Examples:**
- `problem_2x2_0.5_10_3` = 2x2 matrix, 50% sparsity, condition number 10, instance 3
- `problem_16x16_0.9_50_5` = 16x16 matrix, 90% sparsity, condition number 50, instance 5

## 🔧 For Recipients

### **Python Usage**
```python
import numpy as np

# Load a problem
problem_name = "problem_4x4_0.5_10_1"
A = np.load(f"{problem_name}_A.npy")
b = np.load(f"{problem_name}_b.npy")
csol = np.load(f"{problem_name}_csol.npy")

# Verify solution
residual = np.linalg.norm(A @ csol - b)
print(f"Residual: {residual}")
```

### **MATLAB Usage**
```matlab
% Load a problem
problem_name = 'problem_4x4_0.5_10_1';
A = load([problem_name '_A.npy']);
b = load([problem_name '_b.npy']);
csol = load([problem_name '_csol.npy']);

% Verify solution
residual = norm(A * csol - b);
fprintf('Residual: %e\n', residual);
```

## ✅ Quality Assurance

- **All matrices are Hermitian and positive definite**
- **Solutions verified with residuals < 1e-15**
- **Fixed seeds ensure reproducibility**
- **Power-of-2 dimensions required for HHL**
- **Controlled sparsity and condition numbers**

## 📊 Dataset Statistics

| Matrix Size | Instances | Sparsity Range | Condition Number Range |
|-------------|-----------|----------------|----------------------|
| 2x2         | 5         | 0.0 - 0.5      | 3.8 - 41.0           |
| 4x4         | 5         | 0.25 - 0.5     | 4.7 - 41.0           |
| 8x8         | 5         | 0.0 - 0.656    | 2.0 - 50.0           |
| 16x16       | 5         | 0.0 - 0.75     | 2.0 - 50.0           |

## 🎯 Intended Use Cases

1. **HHL Algorithm Testing** - Benchmark quantum linear solvers
2. **Performance Comparison** - Compare different quantum backends
3. **Research Validation** - Test new quantum algorithms
4. **Educational Purposes** - Learn about quantum linear algebra
5. **Reproducible Research** - Standardized test problems

## 📞 Support

- **README.md** contains detailed usage instructions
- **Metadata files** provide complete problem information
- **Test script** (`test_dataset_loading.py`) verifies dataset integrity
- **Source code** available in `Generate_Problem_V2.py`

## 🔄 Reproducibility

All problems use deterministic seeds based on:
- Matrix size
- Target sparsity  
- Target condition number
- Instance number

This ensures identical results across different systems and runs.

---

**Generated**: August 26, 2024  
**Tool**: Generate_Problem_V2.py  
**Status**: ✅ Ready for Distribution
