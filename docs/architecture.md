# Architecture: Readout, Post-Processing, and the Solver Pipeline

This document describes the runtime architecture of the qlsas package — how
a problem `(A, b)` is turned into a classical solution `x`, what the moving
parts are, and how to extend the system with new QLSA algorithms (e.g.
QSVT) or new readout strategies (e.g. shadow tomography).

It is the reference for anyone touching the readout / post-processing /
solver-dispatch path. For the HRF tomography algorithm specifically, see
[hrf_readout.md](hrf_readout.md).

## End-to-end pipeline

```
QLSA.build_circuit(A, b)
    │
    ▼
QLSACircuit { circuit, solution_register, ancilla_register, ancilla_creg, success_criterion, params }
    │                                                          │
    │                                                          │  defines what
    │                                                          │  bitstring pattern
    │                                                          │  marks success
    ▼                                                          │
[Single-circuit readout]               [MultiCircuitReadout]   │
Readout.apply(qlsa_circuit)            Readout.build_circuits  │
   ↓                                       ↓ (returns N+1)     │
QuantumCircuit                         list[QuantumCircuit]    │
   ↓                                       ↓                   │
Transpiler.optimize()                  Transpiler (per circuit)│
   ↓                                       ↓                   │
Executer.run() ───────┐         ┌── Executer.run() (each)      │
                      ↓         ↓                              │
                 MeasurementResult ─────┐                      │
                                        │                      │
                                        ▼                      │
              MeasurementResult.get_postselected_counts(register_names, success_criterion)
                                        │                      │
                                        ▼                      │
[Single-circuit] Readout.process(result, A, b) → TomographyResult
[Multi-circuit ] Readout.combine_results(results, A, b, success_criterion) → TomographyResult
                                        │
                                        ▼
                  SolveResult { solution, direction, alpha, success_rate, residual, metadata }
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                    one-shot caller          Refiner.refine()
                    uses .solution           uses .direction (computes own α)
```

## Components

### `QLSACircuit` ([readout/base.py](../src/qlsas/readout/base.py))

A dataclass that the QLSA returns from `build_circuit`. Carries:

- `circuit` — the pre-readout `QuantumCircuit`.
- `solution_register` — `QuantumRegister` that will hold |x⟩.
- `ancilla_register` / `ancilla_creg` — single-ancilla metadata, retained for HHL's circuit-construction use. New code should treat these as HHL-specific implementation detail and rely on `success_criterion` for post-selection.
- `success_criterion` — a `SuccessCriterion` describing which classical-register values mark a shot as "successful." HHL populates this with one register required to be `"1"`. QSVT solvers populate it with multiple registers and arbitrary required patterns.
- `params` — algorithm-specific computed parameters (e.g. HHL stores `t0`, `C`).

### `SuccessCriterion` ([readout/base.py](../src/qlsas/readout/base.py))

The single source of truth for "what counts as a successful shot."

```python
@dataclass
class SuccessCriterion:
    registers: list[ClassicalRegister]
    required_values: list[str]

    def matches(self, bitstring: str) -> bool: ...
```

Convention: `registers[0]` occupies the **rightmost** characters of the
joined measurement bitstring; `registers[1]` sits to its left, etc.
Readouts list the success registers first in their `register_names`
property so they end up at the LSBs (rightmost) of the joined bitstring.

For HHL: one register, width 1, required `"1"` — equivalent to the legacy
`key[-1] == "1"` rule. For QSVT (when added): multiple registers with
arbitrary success patterns (e.g. `["1", "11"]`).

### `Readout` ([readout/base.py](../src/qlsas/readout/base.py))

Abstract base for **single-circuit** readouts. Concrete subclasses today:

| Class | Purpose | Module |
|---|---|---|
| `MeasureXReadout` | Direct Z-basis measurement of the solution register. | [readout/measure_x.py](../src/qlsas/readout/measure_x.py) |
| `SwapTestReadout` | Estimates `⟨x⏐v⟩` against a reference vector via a swap test. | [readout/swap_test.py](../src/qlsas/readout/swap_test.py) |

Three required methods:

- `register_names` — list of classical-register names to be joined for post-selection. Convention: success-criterion registers first (rightmost in bitstring), then the readout's own measurement registers.
- `apply(qlsa_circuit, *, state_prep=None)` — append measurement / auxiliary gates to the QLSA circuit and return a `QuantumCircuit` ready for transpilation. May stash `qlsa_circuit.success_criterion` on the instance for later use by `process()`.
- `process(result, A, b, verbose=True)` — turn a `MeasurementResult` into a `TomographyResult` (or a swap-test-specific tuple).

### `MultiCircuitReadout` ([readout/base.py](../src/qlsas/readout/base.py))

Marker base for readouts that submit **more than one circuit per solve**.
Today only HRF; future shadow tomography or other `O(N)`-circuit
strategies will subclass this.

```python
class MultiCircuitReadout(Readout):
    @abstractmethod
    def build_circuits(self, qlsa_circuit: QLSACircuit) -> list[QuantumCircuit]: ...

    @abstractmethod
    def combine_results(
        self,
        results: list[MeasurementResult],
        A: np.ndarray,
        b: np.ndarray,
        success_criterion: SuccessCriterion | None = None,
        verbose: bool = True,
    ) -> TomographyResult: ...
```

The solver dispatches on this protocol (`isinstance(self.readout, MultiCircuitReadout)`),
not on any concrete subclass. Adding a new multi-circuit strategy needs
zero solver changes.

### `MeasurementResult` ([measurement_result.py](../src/qlsas/measurement_result.py))

Backend-agnostic wrapper around a Qiskit `SamplerPubResult` (IBM/Aer) or
a plain `dict[str, int]` (Quantinuum / synthetic counts). Exposes:

- `get_counts(register_names)` — joined `{bitstring: count}` dict.
- `get_bitstrings(register_names)` — flat per-shot list.
- `get_postselected_counts(register_names, success_criterion)` — **the only place "what counts as a success shot" is computed.** Returns `(filtered_counts, num_successful, total)`. Falls back to the legacy `key[-1] == "1"` rule when `success_criterion is None`, which keeps tests with synthetic counts working.

There is also a module-level helper `to_counts(result, register_names)`
that normalises any of `MeasurementResult` / `SamplerPubResult` / `dict`
into a counts dict, used by single-circuit readouts.

### `TomographyResult` ([readout/base.py](../src/qlsas/readout/base.py))

Uniform return type for tomography readouts:

```python
@dataclass
class TomographyResult:
    direction: np.ndarray   # always unit-norm
    alpha: float            # least-squares scale via norm_estimation; nan if N/A
    success_rate: float
    residual: float         # ‖b − A·(α·direction)‖
    metadata: dict
```

Iterable as a 3-tuple `(direction, success_rate, residual)` so legacy
tuple-unpacking code keeps working: `solution, success_rate, residual =
readout.process(...)`. New code should access fields by name and use
`.scaled` (= `α · direction`) when the physically-scaled vector is
needed.

### `post_processor` ([post_processor.py](../src/qlsas/post_processor.py))

Module-level free functions (no class wrapper):

- `norm_estimation(A, b, x)` — closed-form `α` minimising `‖A·(α·x) − b‖²`.
- `tomography_from_counts(counts, A, b, success_criterion=None)` — reconstruct a `TomographyResult` from a counts dict. Used by `MeasureXReadout`.
- `swap_test_from_counts(counts, A, b, swap_test_vector, success_criterion=None)` — used by `SwapTestReadout`.
- `_finish_tomography(...)` — internal helper called by `tomography_from_counts`. Performs sign correction, unit-norm assertion, and α/residual computation.

### `QuantumLinearSolver` ([solver.py](../src/qlsas/solver.py))

The orchestrator. `solve(A, b, ...)` does:

1. Call `qlsa.build_circuit(A, b, state_prep)` → `QLSACircuit`.
2. **Dispatch on readout protocol:**
   - If `isinstance(readout, MultiCircuitReadout)` → `_solve_multi`. Calls `build_circuits` → transpile each → execute each → `combine_results` → `TomographyResult` → `SolveResult`.
   - Else → single-circuit path. Calls `readout.apply` → transpile → execute → `readout.process` → `TomographyResult` → `SolveResult`. If `target_successful_shots` is set, repeats execution batches until enough successful shots accumulate.
3. Return `SolveResult`.

The internal helper `_to_solve_result(proc_result, ...)` wraps a
`TomographyResult` into a `SolveResult`, copying through `direction`,
`alpha`, and metadata. It also handles the legacy 3-tuple return from
`SwapTestReadout` (`(value, success_rate, residual)`).

Post-selection in the shot-accumulation paths (`_ibm_successful_shots`,
`_quantinuum_successful_shots`, `_trim_counts_to_target`) uses a single
predicate built from the QLSA's `SuccessCriterion`:

```python
def _success_predicate(success_criterion):
    if success_criterion is None:
        return lambda key: bool(key) and key[-1] == "1"
    return success_criterion.matches
```

### `SolveResult` ([solver.py](../src/qlsas/solver.py))

Value object returned to callers:

- `solution` — physically-scaled vector (`α · direction`). The user-facing API for one-shot callers.
- `direction` — unit-norm reconstructed direction. **Iterative refinement reads this**, then computes its own scale, eliminating any silent dependence on whether the readout pre-scaled the output.
- `alpha` — the least-squares scale.
- `success_rate`, `residual`, `metadata` — diagnostics.

Numpy interop is preserved (`__array__`, `__mul__`, etc. delegate to
`solution`), so `np.asarray(result)`, `alpha * result`, and
`result.shape` keep working.

### `Refiner` ([refiner.py](../src/qlsas/refiner.py))

Iterative-refinement loop built on top of `QuantumLinearSolver`. Per
iteration:

1. Builds a normalised sub-problem `(A/‖new_r‖, new_r/‖new_r‖)`.
2. Calls `solver.solve(...)` to get a `SolveResult`.
3. Reads `solve_result.direction` (unit-norm).
4. Computes its own α via `norm_estimation(A, new_r, new_x)`.
5. Accumulates `x += (α / nabla) * new_x`.

The unit-norm contract enforced by `direction` is the load-bearing
correctness fix from this refactor — see the migration notes below.

## Extending the system

### Adding a new single-circuit readout

1. Subclass `Readout`.
2. Implement `register_names`, `apply(qlsa_circuit, ...)`, `process(result, A, b, ...)`.
3. In `process`, route post-selection through `MeasurementResult.get_postselected_counts(self.register_names, success_criterion)`. Do not write `key[-1] == "1"` filters.
4. Return a `TomographyResult` with `direction` unit-norm and `alpha` set via `norm_estimation`.

### Adding a new multi-circuit readout (e.g. shadow tomography)

1. Subclass `MultiCircuitReadout`.
2. Implement `build_circuits(qlsa_circuit)` returning all circuits (any number) in execution order.
3. Implement `combine_results(results, A, b, success_criterion, verbose)` to reconstruct from per-circuit `MeasurementResult`s. Return a `TomographyResult` with unit-norm `direction`.
4. The solver auto-dispatches on the `MultiCircuitReadout` marker — no solver changes needed.

### Adding a new QLSA algorithm (e.g. QSVT)

1. Subclass `QLSA` ([algorithms/base.py](../src/qlsas/algorithms/base.py)).
2. Implement `build_circuit(A, b, state_prep, **kwargs)` to construct the algorithm circuit.
3. **Construct a `SuccessCriterion`** describing which classical registers must hold which patterns for a shot to be successful. For QSVT this is typically multiple ancilla registers, e.g.:
   ```python
   SuccessCriterion(
       registers=[anc_a, anc_b],
       required_values=["1", "11"],
   )
   ```
4. Return a `QLSACircuit` with the new criterion attached. Existing readouts (MeasureX, HRF) work without modification because all post-selection routes through the criterion.

For QSVT specifically, the `SolveResult.direction` contract means the
existing `Refiner` will work unchanged once a QSVT QLSA exists. The only
work remaining for full QSVT support is the algorithm itself; the
readout/post-processing/refiner pipeline is QSVT-ready.

### Repeat-until-success scaffolding (for low-success-probability QLSAs)

The existing `target_successful_shots` parameter on `QuantumLinearSolver`
already implements RUS-style shot accumulation: it batches executions
until the accumulated count of post-selected-successful shots reaches the
target. It now reads the success predicate from
`qlsa_circuit.success_criterion`, so it is QSVT-aware. If QSVT needs a
different RUS policy (e.g. mid-circuit measurement and reset, fast
classical decision logic between batches), that policy can extend this
machinery without changes to the readout/post-processing layers.

## Conventions

### Bitstring layout

Per Qiskit `SamplerPubResult.join_data` semantics:

> The first name in `register_names` becomes the **rightmost** characters
> (LSB) of the joined bitstring; the last name becomes the leftmost.

Therefore each readout puts the **success-criterion registers first** in
its `register_names`, so success bits land at the rightmost positions of
the bitstring. The readout's own measurement registers come after, ending
up at the MSBs.

`SuccessCriterion.matches(bitstring)` exploits this: it walks
`registers[0..N]` and reads the rightmost bits of the bitstring inward.

### Solution bit extraction

After post-selection, the solution-register bits are always the
**leftmost** `log2(len(b))` characters of the bitstring. Used by
`tomography_from_counts` and `HRFReadout._postselect_probs`. This is
robust to any number of success registers as long as they're all at the
rightmost positions.

### Norm estimation (the two call sites)

`norm_estimation` is called in two distinct places with different semantics:

- **In tomography** ([post_processor.py](../src/qlsas/post_processor.py)) — α computed against the (A, b) the solver was given. Used to populate `TomographyResult.alpha` for diagnostic / one-shot use.
- **In iterative refinement** ([refiner.py](../src/qlsas/refiner.py)) — α computed against `(A, new_r)` for the **current** sub-problem. Used to accumulate the iterate.

The two are independent because IR feeds the solver a *normalised* sub-problem
(`A_normalized = A/‖new_r‖`, RHS `= new_r/‖new_r‖`), so the α the readout
computes is for the sub-problem, not the global problem; IR has to recompute α
against the original A and the unnormalised residual. IR uses
`solve_result.direction` (unit-norm) so its own α is the *only* scale applied
to the iterate, eliminating the prior silent dependence on whether the readout
pre-scaled the vector.

## What changed in this refactor (migration notes)

The refactor was driven by two pressures: (1) HRF integration exposed
abstraction cracks (special-cased solver dispatch, three different
`process()` contracts, hidden state in `HRFReadout`, four duplicated
post-selection sites), and (2) upcoming QSVT solvers will use
multi-register success criteria, which the previous `key[-1] == "1"`
hardcode does not support.

### New public APIs

| API | Purpose |
|---|---|
| `qlsas.readout.base.SuccessCriterion` | Describes the success-bit pattern. Lives on `QLSACircuit.success_criterion`. |
| `qlsas.readout.base.TomographyResult` | Uniform return type for tomography readouts. Iterable as the legacy 3-tuple. |
| `qlsas.readout.base.MultiCircuitReadout` | Protocol for readouts that submit > 1 circuit per solve. |
| `MeasurementResult.get_postselected_counts(register_names, success_criterion)` | Single source of truth for shot post-selection. |
| `qlsas.measurement_result.to_counts(result, register_names)` | Centralised helper, used by single-circuit readouts. |
| `SolveResult.direction`, `SolveResult.alpha` | Unit-norm direction + least-squares scale, alongside the existing `.solution` (= `direction * alpha`). |

### Removed

- `qlsas.post_processor.Post_Processor` class. Its methods are now module-level functions: `norm_estimation`, `tomography_from_counts`, `swap_test_from_counts`, `_finish_tomography`. Calling code that did `Post_Processor().tomography_from_counts(...)` becomes `tomography_from_counts(...)`.
- The `post_processor` keyword argument on `MeasureXReadout` and `SwapTestReadout`. Both readouts now import the module-level functions directly.
- The `_solve_hrf` method on `QuantumLinearSolver`, replaced by the generic `_solve_multi`.
- The `isinstance(self.readout, HRFReadout)` dispatch in `QuantumLinearSolver.solve()`, replaced by `isinstance(self.readout, MultiCircuitReadout)`.
- The duplicated `_to_counts` helper that lived in both `measure_x.py` and `swap_test.py`. Replaced by `qlsas.measurement_result.to_counts`.

### Behaviour changes

- **`HRFReadout.process()` now returns the unit-norm direction**, not the pre-scaled solution. The `TomographyResult` carries `alpha` separately. Callers that read `.solution` from `SolveResult` continue to receive the scaled vector (computed as `α · direction`); callers that read `.direction` (i.e. `Refiner`) get the unit-norm vector. This eliminates a latent correctness risk where `Refiner` produced numerically correct iterates only because `norm_estimation` happens to be invariant under input scaling.
- **Post-selection across the whole codebase routes through `SuccessCriterion`.** The legacy `key[-1] == "1"` rule is preserved as a fallback when `success_criterion is None` (so tests with synthetic counts still work), but production paths populate the criterion and use `SuccessCriterion.matches`.

### Backward compatibility

The refactor preserved every public surface that had test coverage. In particular:

- `tomography_from_counts(counts, A, b)` (without `success_criterion`) still works and uses the legacy fallback.
- `Readout.process(result, A, b)` callers that did `solution, success_rate, residual = readout.process(...)` keep working — `TomographyResult` is iterable as a 3-tuple.
- `SolveResult.solution` keeps returning the scaled vector. Existing `np.asarray(result)`, `alpha * result`, etc. continue to work via the dataclass's numpy-interop dunders.
- `HRFReadout.apply()`, `HRFReadout.build_hrf_circuits()`, `HRFReadout._extract_probs()`, `HRFReadout.process()` all still work — they are now thin shims around the canonical `build_circuits()` / `combine_results()` API. The HRF unit tests that assert on stashed instance state (`_base_circuit_core`, etc.) still pass.

A follow-up cleanup PR can delete the HRF backward-compat shims once
the ~6 tests that depend on internal state are migrated to the canonical
API.

### Migration checklist for downstream code

If your code …                                          | Do this
:------------------------------------------------------|:--------
imports `Post_Processor`                               | Import the module-level functions instead: `from qlsas.post_processor import norm_estimation, tomography_from_counts, swap_test_from_counts`.
constructs a `Readout` with `post_processor=...`       | Drop the kwarg; readouts use the module functions directly.
defines a custom `Readout` subclass                    | Have `process()` return a `TomographyResult` (or a tuple — both work). Route post-selection through `MeasurementResult.get_postselected_counts(self.register_names, success_criterion)` instead of inlining a bitstring filter.
defines a custom `QLSA` subclass                       | Construct a `SuccessCriterion` and attach it to the returned `QLSACircuit`. For a single-ancilla success-on-one design, that's `SuccessCriterion(registers=[anc_creg], required_values=["1"])`.
reads `solve_result.solution` from `QuantumLinearSolver` | No change. Still returns the scaled vector.
writes a custom iterative refiner / RUS loop           | Read `.direction` (unit-norm) and compute your own scale via `norm_estimation`. Don't depend on the readout pre-scaling its output — that contract no longer holds.

## Test surface

- 178 tests pass against `AerSimulator` + `hadamard-random-forest`.
- New architecture tests are in [tests/test_architecture_refactor.py](../tests/test_architecture_refactor.py) — covers single + multi-register `SuccessCriterion`, `get_postselected_counts`, and a synthetic `MultiCircuitReadout` proving generic dispatch.
- The HRF integration test `test_hrf_vs_measure_x_agreement` ([tests/test_hrf_readout.py](../tests/test_hrf_readout.py)) verifies that HRF and MeasureX agree on direction (cosine similarity > 0.8 across 8192 shots) — this is the regression net for the unit-norm-direction change in `HRFReadout.process()`.

## What's not addressed by this refactor

- **The Quantinuum / IBM dispatch in the executer/solver** — orthogonal concern, not touched.
- **Resource estimation, transpilation, error mitigation** — orthogonal, not touched.
- **The `MeasureXReadout` "cheat" that calls `LA.solve` for sign correction** — by design (it's a benchmarking readout). For honest end-to-end quantum behaviour, use `HRFReadout`.
- **Repeat-until-success policy enrichment** — the existing `target_successful_shots` machinery already handles RUS-style shot accumulation and now reads from `success_criterion`. Any QSVT-specific RUS extensions can build on that.
