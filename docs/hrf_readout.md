# Hadamard Random Forest Readout

`HRFReadout` integrates the Hadamard Random Forest (HRF) quantum state tomography scheme
([arXiv:2505.06455](https://arxiv.org/abs/2505.06455), Bhole et al. 2025) into the QLSA
readout framework. It is a drop-in replacement for `MeasureXReadout` that reduces the
number of measurement circuits from **O(3^N) → O(N)**.

---

## Background: why standard tomography is expensive

Standard quantum state tomography (QST) reconstructs an N-qubit state by measuring in
all 3^N Pauli bases ({X, Y, Z}^⊗N). For HHL on a k-bit solution, that is 3^k circuits
per solve call — prohibitive even for k = 10.

`MeasureXReadout` sidesteps this by measuring only in the Z basis and using the known
classical solution to correct the signs of the recovered amplitudes. This works, but it
defeats the purpose of quantum advantage for large problems where the classical solution
is too expensive to compute.

`HRFReadout` recovers signs **quantumly**, using only N + 1 circuits:

| Readout         | Circuits per solve | Sign recovery       |
|-----------------|-------------------|---------------------|
| `MeasureXReadout` | 1               | Classical (uses `LA.solve`) |
| `HRFReadout`    | N + 1             | Quantum (HRF algorithm) |
| Full QST        | 3^N               | Quantum              |

---

## How HRF works

### Step 1 — Amplitude recovery (base circuit)

The base HHL circuit is executed and the solution register is measured in the
computational (Z) basis. Conditioned on the ancilla flag being 1, the probability of
measuring basis state |k⟩ is:

```
P(|k⟩) = |x_k|²
```

This gives the **magnitudes** of all amplitudes but loses their signs.

### Step 2 — Sign recovery (Hadamard circuits)

For each solution qubit i (i = 0, …, N−1), a second circuit is executed in which a
Hadamard gate is applied to qubit i before the Z-basis measurement. The resulting
probability distribution carries cross-terms between adjacent amplitudes:

```
P_H(|k⟩ | H on qubit i) = ½ |x_k ± x_{k ⊕ 2^i}|²
```

The relative sign between x_k and x_{k ⊕ 2^i} (states that differ only in bit i)
determines whether the ± is + or −.

From these two distributions the relative sign can be recovered as:

```
sign(x_root, x_leaf) = sign( 2·P_H(|root⟩) − P(|root⟩) − P(|leaf⟩) )
```

### Step 3 — Global sign propagation via Random Forest

The N-dimensional computational basis can be viewed as the vertices of an N-dimensional
hypercube: two states are adjacent if they differ in exactly one bit. The relative sign
between adjacent vertices is computed in Step 2.

To reconstruct all 2^N global signs from N local comparisons, HRF generates a random
spanning tree of the hypercube. Propagating signs along this tree from a fixed root
gives a full sign assignment. Since any spanning tree covers all vertices, one tree is
enough in principle — but shot noise can corrupt individual comparisons.

HRF uses **majority voting over `num_trees` independent random spanning trees**:

```
global_sign(k) = majority_vote( {tree_sign_j(k)}_{j=1}^{num_trees} )
```

More trees → more robust sign estimates at the cost of classical post-processing time.
The classical cost per tree is O(2^N) (proportional to the Hilbert space), so the
computational bottleneck remains exponential in N, as it must be for full state
reconstruction.

### Step 4 — Solution reconstruction

The reconstructed statevector is:

```
|x̂⟩ = amplitudes ⊙ signs = √P(|k⟩) · global_sign(k)  for each k
```

It is normalized to unit norm, then physically scaled using `norm_estimation` so that
`α·|x̂⟩` minimizes ‖A·(α·|x̂⟩) − b‖².

---

## Validity assumption

HRF is designed for **real-valued** quantum states. HHL always produces a real solution
when A is real symmetric and b is real, because:

1. The Hamiltonian simulation `exp(−iAt)` maps a real initial state to a complex one,
   but the QPE and eigenvalue inversion are constructed so that the final state
   `|x⟩ ∝ Σ_i λ_i⁻¹ ⟨u_i|b⟩ |u_i⟩` has real amplitudes in the computational basis
   whenever A has a real eigenvector basis (which all real symmetric matrices do) and
   b is real.

2. The HRF sign formula `sign(2P_H(r) − P(r) − P(l))` computes the _real_ relative
   sign between adjacent amplitudes. If the state were genuinely complex, the formula
   would give the sign of the real part of the cross-term, which is incorrect.

In practice, HHL produces amplitudes with imaginary parts at the ≈1e-15 level
(floating-point noise). `HRFReadout.process()` discards the imaginary component via
`.real` before normalizing.

---

## Usage

### Installation

Install the optional HRF dependency alongside the qlsas package:

```bash
pip install "qlsas[hrf]"
# or, until it is on PyPI:
pip install git+https://github.com/comp-physics/Quantum-HRF-Tomography.git
pip install treelib mthree
```

### Basic usage — swap `MeasureXReadout` for `HRFReadout`

```python
from qlsas.solver import QuantumLinearSolver
from qlsas.algorithms.hhl import HHL, MCRYEigOracle
from qlsas.readout import HRFReadout          # ← the only change
from qlsas.state_prep import DefaultStatePrep
from qiskit_aer import AerSimulator
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 0.0])  # will be auto-normalised inside HHL

solver = QuantumLinearSolver(
    qlsa=HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle()),
    readout=HRFReadout(num_trees=20),   # ← HRF instead of MeasureXReadout()
    backend=AerSimulator(),
    state_prep=DefaultStatePrep(),
    shots=4096,
)

result = solver.solve(A, b / np.linalg.norm(b))
print(result.solution)        # solution vector
print(result.success_rate)    # mean ancilla success rate across all N+1 circuits
print(result.residual)        # ‖A·x − b‖
print(result.metadata)        # {"num_hrf_circuits": 2, "num_trees": 20}
```

### Tuning `num_trees`

| Setting          | When to use                                              |
|------------------|----------------------------------------------------------|
| `num_trees=5–10` | Quick prototyping, simulators with many shots            |
| `num_trees=20`   | Default; good balance of robustness and compute time     |
| `num_trees=50+`  | Real hardware with significant shot noise; large N       |

Classical post-processing time scales as O(num_trees × 2^N), so very large values are
only worthwhile when sign errors from shot noise are the dominant error source.

---

## Circuit structure

For a 2-qubit solution register (4×4 system), `HRFReadout` produces 3 circuits:

```
Circuit 0 (base):
  [HHL core] → measure ancilla_flag → inverse QPE → measure hrf_x_result

Circuit 1 (H on qubit 0):
  [HHL core] → measure ancilla_flag → inverse QPE → H(q0) → measure hrf_x_result

Circuit 2 (H on qubit 1):
  [HHL core] → measure ancilla_flag → inverse QPE → H(q1) → measure hrf_x_result
```

The ancilla is measured mid-circuit at step 4 of HHL (standard Qiskit dynamic-circuit
feature). The Hadamard and solution measurement are appended after the full HHL
uncomputation (inverse QPE), so they act on the post-selected solution state.

---

## SolveResult fields

| Field            | Meaning                                                  |
|------------------|----------------------------------------------------------|
| `solution`       | Reconstructed and physically-scaled solution vector       |
| `success_rate`   | Mean ancilla flag success rate across all N+1 circuits   |
| `residual`       | ‖A·solution − b‖                                        |
| `metadata["num_hrf_circuits"]` | N+1 (1 base + N H-variants)              |
| `metadata["num_trees"]`        | `num_trees` used for majority voting         |

---

## Limitations

- **Real solutions only.** Complex HHL solutions (e.g., if A has complex entries or b
  is complex) require a different sign-recovery approach.
- **`target_successful_shots` is unsupported.** HRF requires a fixed shot budget per
  circuit; accumulating until a target ancilla count is hit would require re-running the
  full N+1-circuit suite multiple times and combining the distributions non-trivially.
- **Quantinuum backends are not yet supported.** Only IBM/Aer backends work currently.
- **Classical post-processing is still exponential.** HRF reduces quantum measurement
  overhead to O(N), but the classical reconstruction step (running over all 2^N basis
  states) remains exponential. The break-even point where HRF is faster than QST is
  roughly N ≈ 10–15 depending on hardware gate time and shot count.

---

## References

- Bhole et al., *"Hadamard Random Forest: Real-Valued Quantum State Tomography with
  Exponentially Fewer Measurements"*, arXiv:2505.06455 (2025).
- Source code: <https://github.com/comp-physics/Quantum-HRF-Tomography>
