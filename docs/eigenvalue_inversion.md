# Eigenvalue inversion in HHL

The HHL algorithm encodes the spectrum of a Hermitian matrix `A` into a QPE register and then applies, conditioned on each eigenvalue estimate, an RY rotation on a single ancilla qubit so that the success-branch amplitude is proportional to `1/λ_j`.  This step — the *eigenvalue inversion oracle* — is what makes HHL work, and it is also the dominant cost contributor of the algorithm in many regimes.

This package ships three oracle implementations.  They all consume the same `(t0, C, A)` parameters and produce the same logical operation in the success branch, but they differ substantially in:

- circuit depth and gate count,
- behaviour on out-of-range / boundary QPE states,
- and which use cases they are best suited for.

This document is the canonical reference for those trade-offs.

---

## 1. What every oracle is doing

After forward QPE, the state on the `1 + m + n` qubit register (ancilla, QPE register, data register) is approximately

```
|ψ⟩  =  Σ_j  α_j  |0⟩_anc ⊗ |λ̂_j⟩_qpe ⊗ |u_j⟩_data
```

where `α_j = ⟨u_j|b⟩` and `|λ̂_j⟩` is the QPE encoding of phase `φ_j = λ_j · t0 / (2π)`.

The oracle's job is to apply the controlled rotation

```
|0⟩_anc ⊗ |k⟩_qpe   ↦   ( cos(θ_k/2) |0⟩ + sin(θ_k/2) |1⟩ )_anc ⊗ |k⟩_qpe
```

with rotation angle

```
θ_k = 2 · arcsin( C / λ_k )      for every QPE basis state k
```

and the implied eigenvalue

```
λ_k = 2π · φ_k / t0,   where φ_k = k/2^m  (with two's-complement unwrap if A is indefinite).
```

State `k=0` is left untouched (`φ=0` ⇒ `λ=0`, no inversion possible).  After the rotation, measuring the ancilla in `|1⟩` projects onto a state whose data register is proportional to `A⁻¹|b⟩`.

So the oracle is fully specified by the angle table `θ_k`.  All three oracles in this package compute the same `θ_k` (to within boundary-state edge cases noted below).  They differ only in **how** they realise the controlled-rotation circuit.

---

## 2. The three oracles

### 2.1 `MCRYEigOracle` — multi-controlled RY per state

**Algorithm.**  Loop over `k = 1, …, 2^m − 1`.  For each `k`, build an `m`-controlled RY gate whose control state is the bitstring of `k` and whose rotation angle is `θ_k`.  Append all `2^m − 1` gates in series.

**Implementation:** `mcry_eig_inversion` in [`hhl_helpers.py`](../src/qlsas/algorithms/hhl/hhl_helpers.py).

**Complexity** (after Qiskit's standard MCRY decomposition):

| Quantity | Value |
|---|---|
| Gate count | `O(m · 2^m)` |
| Depth | `O(m · 2^m)` |
| Ancillas | 0 |
| Connectivity | All-to-all (each MCRY reaches every QPE qubit) |

**Pros.**

- Conceptually simplest — one explicit gate per QPE state.
- Easiest to debug: you can inspect any single `θ_k` rotation by name.
- Same unitary as `UCRYEigOracle` on every QPE state, so it's a useful reference oracle.

**Cons.**

- Strictly worse depth than `UCRYEigOracle` by a factor of `m`.
- Multi-controlled gates are expensive to transpile to native gate sets and fault-tolerant primitives.

**When to choose it.**  Reference / debugging only.  It has no production-use advantage over `UCRYEigOracle`.

---

### 2.2 `UCRYEigOracle` — Möttönen uniformly-controlled RY tree (default)

**Algorithm.**  Compute the full angle table `θ_0, …, θ_{2^m−1}` once.  Then realise the table as a single uniformly-controlled RY using the Möttönen recursion:

```
UCR_n(θ)  =  UCR_{n−1}(α)  ·  CX(ctrl_{n−1}, target)  ·  UCR_{n−1}(β)  ·  CX(ctrl_{n−1}, target)
```

where `α_k = (θ_k + θ_{k+half})/2` and `β_k = (θ_k − θ_{k+half})/2`, and `half = 2^{n−1}`.  At the leaves (`n=0`) emit a single RY with the leaf angle.

**Implementation:** `ucry_eig_inversion` in [`hhl_helpers.py`](../src/qlsas/algorithms/hhl/hhl_helpers.py); the Möttönen recursion lives in `_apply_ucry`.

**Complexity:**

| Quantity | Value |
|---|---|
| RY gates | `2^m` |
| CX gates | `2^{m+1} − 2` |
| Depth | `O(2^m)` |
| Ancillas | 0 |
| Connectivity | Linear chain on QPE qubits + CX to target |

**Pros.**

- **Strictly better than MCRY**: identical unitary, factor-`m` shallower.
- **Numerically robust on out-of-range states.**  When `|C/λ_k| > 1`, the implementation clamps to `±1` (i.e. `θ_k = ±π`, full ancilla flip).  This is approximately correct on QPE-leakage states and well-behaved for iterative refinement (see §4).
- **Handles indefinite-spectrum boundary cleanly.**  For matrices with negative eigenvalues, the most-negative-phase state `k = 2^{m−1}` produces a real rotation (`θ = 2·arcsin(±C·t0/π)`) — there's no special case.
- **Identical unitary to `MCRYEigOracle`** on every QPE state, so it can be used as a drop-in replacement and verified against MCRY for small `m`.

**Cons.**

- Slightly deeper than `ExactReciprocalEigOracle` (~10–20% constant factor) because Qiskit's internal `UCRYGate` reorders the angle table to merge some adjacent CXs.
- The recursive computation of the angle table is `O(2^m)` classical work; this is the same as the other oracles but worth noting if `m` is very large.

**When to choose it.**  This is the recommended default.  Use `UCRYEigOracle` unless you have a specific reason to pick something else.

---

### 2.3 `ExactReciprocalEigOracle` — Qiskit's `ExactReciprocalGate`

**Algorithm.**  Construct a Qiskit `ExactReciprocalGate(num_state_qubits=m, scaling=S, neg_vals=indefinite)` and append it to the circuit.  Internally the gate

1. computes its own angle table from `S` and `m`,
2. wraps that table into a `UCRYGate`,
3. (when `neg_vals=True`) adds a second sign-bit-controlled UCRY to handle two's-complement phases.

The `UCRYGate` then decomposes to a Möttönen tree on transpilation.

**Implementation:** `exact_reciprocal_eig_inversion` in [`hhl_helpers.py`](../src/qlsas/algorithms/hhl/hhl_helpers.py).

**Scaling subtlety.**  Qiskit's gate computes angles as `2·arcsin(S · nl / i)` with `nl = 2^m` when `neg_vals=False` and `nl = 2^{m−1}` when `neg_vals=True`.  Matching `S · nl / i = C / λ_k` therefore requires:

```
S = C · t0 / (2π)     when neg_vals = False  (SPD problems)
S = C · t0 / π        when neg_vals = True   (indefinite problems)
```

The implementation handles both cases transparently.

**Complexity:**

| Quantity | Value |
|---|---|
| RY gates | `2^m` (after full decomposition) |
| CX gates | `~2^m` |
| Depth | `O(2^m)` |
| Ancillas | 0 |
| Connectivity | Same as `UCRYEigOracle` — Qiskit's UCRY layout |

So **asymptotically identical to `UCRYEigOracle`**, with a ~10–20% constant-factor depth advantage.

**Pros.**

- Shortest decomposed depth of the three oracles by a constant factor.
- Off-the-shelf — relies on Qiskit's tested implementation.
- One gate at the top level → cleaner `circuit.draw()` output.

**Cons (the structural issues).**

The Qiskit gate has two hard-coded behaviours that cannot be controlled by `S`:

1. **Out-of-range drop.**  When `|S · nl / i| > 1` the angle for state `i` is set to **0** instead of clamping to `π`.  In a saturating regime this *removes* leakage amplitude on small-phase states instead of over-amplifying it.  For `UCRYEigOracle` the same condition triggers a `±π` clamp; for HHL this is closer to the ideal rotation, especially for iterative refinement.
2. **Boundary-state hole.**  When `neg_vals=True`, the gate's `angles_neg[0]` is hard-coded to 0, so the most-negative-phase QPE state `k = 2^{m−1}` is never rotated, regardless of `S`.  `MCRYEigOracle` and `UCRYEigOracle` rotate this state correctly.

These quirks are tracked in [`hhl_helpers.py`](../src/qlsas/algorithms/hhl/hhl_helpers.py) and the `TestQuantumOracleScaling` suite in [`tests/test_hhl_helpers.py`](../tests/test_hhl_helpers.py).

**When to choose it.**  Two scenarios:

- You're operating in the **well-resolved regime** (`m ≫ log₂(κ)`) where saturation never triggers and the boundary state has negligible amplitude — and you want the smallest possible inversion-oracle depth.
- You're testing or instrumenting a pipeline that specifically requires Qiskit's `ExactReciprocalGate` (e.g. comparing to a reference Qiskit HHL implementation).

In other regimes prefer `UCRYEigOracle`.

---

## 3. Side-by-side cheat sheet

| Property | `MCRYEigOracle` | `UCRYEigOracle` *(default)* | `ExactReciprocalEigOracle` |
|---|---|---|---|
| Decomposed depth | `O(m · 2^m)` | `O(2^m)` | `O(2^m)`, ~10–20% smaller constant |
| Same unitary as MCRY? | yes (reference) | **yes** | yes on populated states; differs at boundary `k=2^{m−1}` for indefinite |
| Out-of-range `|C/λ| > 1` | clamps to `±π` | clamps to `±π` | **drops to 0** |
| Indefinite boundary state | rotates correctly | rotates correctly | **never rotated** |
| Ancilla qubits | 0 | 0 | 0 |
| Implementation source | this package | this package | Qiskit |
| Recommended for | reference / debugging | **default** | only when constant-factor depth matters and saturation/boundary do not |

---

## 4. How iterative refinement (IR) changes the picture

A single HHL solve has worst-case relative error `~ κ / 2^m` from QPE phase quantisation.  To hit a target precision `ε` directly, single-shot HHL needs

```
m_single  ~  log₂( κ / ε )
```

i.e. roughly 11–17 QPE qubits for `ε = 10⁻³` on `κ ∈ [2, 100]`.

Iterative refinement does much better.  Solve `Ax_0 ≈ b` to *low* precision, classically compute the residual `r_0 = b − A·x_0`, solve `A·d_0 ≈ r_0` to the same low precision, update `x ← x + d`, repeat.  If each per-solve relative error is `ε_solve < 1`, then after `k` iterations the residual has shrunk by `ε_solve^k`.  So:

```
m_IR  ~  log₂( κ / ε_solve ),     k  ~  log( ε / ε_solve ) / log( ε_solve )
```

with `ε_solve ≈ 0.5` a common rule of thumb.  Concretely:

| κ | `m` (single-shot, ε=10⁻³) | `m` (IR, ε_solve≈0.5) | iters to 10⁻³ |
|---|---|---|---|
| 2  | ~11 | ~3 | ~10 |
| 10 | ~14 | ~5 | ~10 |
| 20 | ~15 | ~6 | ~10 |
| 100 | ~17 | ~8 | ~10 |

**The IR operating point sits at `m ≈ log₂(κ) + 2`** — exactly the regime where the saturating QPE states from §2.3 carry non-negligible amplitude and where `ExactReciprocalEigOracle`'s drop-to-zero behaviour costs the most.

This makes the choice of oracle for IR concrete:

- **Use `UCRYEigOracle` for IR.**  Its clamp behaviour produces `ε_solve ≈ 0.1`–`0.3` in the saturating regime — well within IR's `< 1` requirement, so refinement converges at the expected geometric rate.
- **Avoid `ExactReciprocalEigOracle` for IR at low `m`.**  Drop-to-zero can leave `ε_solve` near `0.5`, occasionally above; refinement either converges very slowly or stalls.
- **`MCRYEigOracle` would also work**, but is dominated by `UCRYEigOracle` on cost.

The depth advantage of `ExactReciprocalEigOracle` over `UCRYEigOracle` (~15% at low `m`) is dwarfed by even one extra IR iteration that bad rotations would cause.

### Running IR in this package

Iterative refinement is implemented as `Refiner` in [`src/qlsas/refiner.py`](../src/qlsas/refiner.py).  It wraps a `QuantumLinearSolver` and handles the residual rescaling, alpha estimation, and convergence check internally — you only need to supply the base solver, the matrix, and the right-hand side:

```python
from qlsas.solver import QuantumLinearSolver
from qlsas.algorithms.hhl import HHL                # default UCRYEigOracle
from qlsas.readout import MeasureXReadout
from qlsas.refiner import Refiner

# Pick m for the *per-iteration* solve, not the target precision:
m_iter = int(np.ceil(np.log2(np.linalg.cond(A)))) + 2

base = QuantumLinearSolver(
    qlsa=HHL(num_qpe_qubits=m_iter),
    readout=MeasureXReadout(),
    backend=backend,
    target_successful_shots=int(1e3),
)
result = Refiner(A, b, solver=base).refine(precision=1e-3, max_iter=20)
# result["x_list"][-1] is the refined solution, result["residuals"] is per-iteration history
```

For a worked sweep over per-iteration `m` and IR iterations on a single problem, see [`examples/experiments.ipynb`](../examples/experiments.ipynb).

---

## 5. Why `MCRY` and `UCRY` are identical and `ExactReciprocal` is *almost* identical

It's worth stating clearly: **all three oracles are different decompositions of the same target unitary up to edge cases.**

- `MCRYEigOracle` realises the angle table by emitting one MCRY per state.  Brute force.
- `UCRYEigOracle` realises the same angle table via Möttönen's recursion.  After full decomposition this is a tree of `2^m` RY and `2^{m+1}−2` CX gates, with the leaf angles being a linear transform of the `θ_k`.
- `ExactReciprocalEigOracle` is a Qiskit gate that *also* internally computes an angle table and *also* hands it off to a `UCRYGate`, which Qiskit then decomposes via the same Möttönen recursion.

So `UCRYEigOracle` and `ExactReciprocalEigOracle` are essentially the same algorithm with two different angle-table builders.  The constant-factor depth difference (`UCRY` ~15% deeper) is from Qiskit's angle ordering tricks and does not represent an asymptotic advantage.

The boundary-state and out-of-range divergences are not algorithmic differences either — they're choices Qiskit's gate makes about what to do when the ideal rotation is undefined.  Our `UCRYEigOracle` makes different (and, for HHL, better) choices: clamp instead of drop, rotate the boundary instead of skipping it.

A *genuinely* asymptotically shallower oracle would require a different algorithm — for example block-encoding `A⁻¹` via QSVT, or applying unary-iteration with prefix-tree ancillas (Babbush et al., 2018).  Those do not live in this package today.

---

## 6. Decision guide

Pick:

- **`UCRYEigOracle`** for everything by default.  It's the cheapest correct oracle, no edge cases bite at any `m`, and IR works cleanly with it.
- **`MCRYEigOracle`** when you specifically want one named `c^m-RY` per QPE state for reference, debugging, transpiler instrumentation, or papers that want to compare against a "naive" baseline.
- **`ExactReciprocalEigOracle`** when (a) you're well-resolved (`m ≫ log₂(κ)`) so saturation and the boundary state have negligible amplitude, and (b) the constant-factor depth saving is worth giving up the cleaner edge-case behaviour.

If you are unsure, choose `UCRYEigOracle`.

---

## 7. References and code pointers

- Möttönen, V., Vartiainen, J. J., Bergholm, V., Salomaa, M. M.  *Quantum Circuits for General Multiqubit Gates*.  Phys. Rev. Lett. **93**, 130502 (2004).  arXiv:[quant-ph/0404089](https://arxiv.org/abs/quant-ph/0404089).
- Harrow, A. W., Hassidim, A., Lloyd, S.  *Quantum algorithm for linear systems of equations*.  Phys. Rev. Lett. **103**, 150502 (2009).  arXiv:[0811.3171](https://arxiv.org/abs/0811.3171).
- Babbush, R. et al.  *Encoding electronic spectra in quantum circuits with linear T complexity*.  Phys. Rev. X **8**, 041015 (2018).  arXiv:[1805.03662](https://arxiv.org/abs/1805.03662) — for the unary-iteration scheme alluded to as a future direction.
- Qiskit `ExactReciprocalGate` source: `qiskit/circuit/library/arithmetic/exact_reciprocal.py`.

Code in this package:

- Oracle classes: [`src/qlsas/algorithms/hhl/eig_oracles.py`](../src/qlsas/algorithms/hhl/eig_oracles.py)
- Implementations: [`src/qlsas/algorithms/hhl/hhl_helpers.py`](../src/qlsas/algorithms/hhl/hhl_helpers.py)
- HHL plumbing: [`src/qlsas/algorithms/hhl/hhl.py`](../src/qlsas/algorithms/hhl/hhl.py)
- Tests covering the trade-offs above: [`tests/test_hhl_helpers.py`](../tests/test_hhl_helpers.py) (in particular `TestQuantumOracleScaling` for the boundary / out-of-range behaviour)
- Empirical sweep across 4×4 and 8×8 instances: [`examples/oracle_sweep_4x4.ipynb`](../examples/oracle_sweep_4x4.ipynb)
- Algorithmic comparison notebook: [`examples/eigenvalue_inversion_comparison.ipynb`](../examples/eigenvalue_inversion_comparison.ipynb)
