# Quantum Linear Solver Abstractions

## UNFINISHED

# Purpose: run circuits on backends, process results into normalized vector proportional to $\vec{x}$, perform tomography to estimate $\vec{x}$.

This directory contains backend-aware solver classes for running quantum
linear system algorithms (QLSAs) and post-processing the results into classical
solution vectors.

## Modules
- `solver.py` – base `Solver` interface and `SolverResult` container.
- `factory.py` – helper for instantiating the right solver given a backend handle or platform hint.

## Usage
```python
import numpy as np
from solvers import create_solver

A = np.array([[1, 0], [0, 2]], dtype=float)
b = np.array([1, 0], dtype=float)

solver = create_solver(platform="qiskit", shots=2048)
result = solver.solve(A, b)
print("solution:", result.vector)
print("success probability:", result.metadata["success_probability"])
```

For Quantinuum backends:
```python
from solvers import create_solver

solver = create_solver(
    backend_name="H2-1E",
    platform="quantinuum",
    shots=1000,
    noisy=True,
)
result = solver.solve(A, b)
```