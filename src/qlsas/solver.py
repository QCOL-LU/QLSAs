from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from qiskit.providers.backend import BackendV2

import numpy as np

from qlsas.quantinuum_config import QuantinuumBackendConfig
from qlsas.executer import Executer
from qlsas.ibm_options import IBMExecutionOptions
from qlsas.algorithms.base import QLSA
from qlsas.state_prep import StatePrep, DefaultStatePrep
from qlsas.readout.base import (
    MultiCircuitReadout,
    QLSACircuit,
    Readout,
    SuccessCriterion,
    TomographyResult,
)
from qlsas.measurement_result import MeasurementResult
from qlsas.transpiler import Transpiler


@dataclass
class SolveResult:
    """Value object returned by :meth:`QuantumLinearSolver.solve`.

    Attributes:
        solution: Physically-scaled solution vector (``alpha * direction``)
            for one-shot callers. Equal to :attr:`direction` when *alpha* is
            unavailable.
        direction: Unit-norm reconstructed solution direction. Iterative
            refinement consumes this and computes its own scale, so swapping
            readouts (MeasureX vs HRF) cannot leak a hidden double-scaling.
        alpha: Least-squares scale that fits ``A·(α·direction) ≈ b``.
            ``None`` when not available.
        success_rate: Fraction of shots that passed the success criterion
            (``None`` when batching is not used).
        residual: Norm of the residual ``‖Ax − b‖`` (``None`` when the caller
            does not supply *A* and *b*).
        metadata: Any additional algorithm-specific diagnostics.
    """

    solution: np.ndarray
    success_rate: Optional[float] = None
    residual: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    direction: Optional[np.ndarray] = None
    alpha: Optional[float] = None

    def __post_init__(self) -> None:
        # When the caller didn't separately supply a direction, fall back to
        # the solution vector so .direction is never silently None for
        # callers that expect a vector (e.g. iterative refinement).
        if self.direction is None:
            self.direction = self.solution

    # --- numpy interop ---------------------------------------------------
    # Lets existing code do  ``alpha * result``  or  ``result.shape``
    # without knowing about SolveResult.

    def __array__(self, dtype=None):
        return np.asarray(self.solution, dtype=dtype)

    def __getattr__(self, name):
        # Delegate unknown attribute access to the underlying array so that
        # e.g. result.shape, result.dtype work transparently.
        try:
            return getattr(self.solution, name)
        except AttributeError:
            raise AttributeError(
                f"'SolveResult' object has no attribute {name!r}"
            ) from None

    def __mul__(self, other):
        return self.solution * other

    def __rmul__(self, other):
        return other * self.solution

    def __add__(self, other):
        return self.solution + other

    def __radd__(self, other):
        return other + self.solution


class QuantumLinearSolver:
    """End-to-end runner for a QLSA: build → readout → transpile → execute → post-process.

    The three algorithmic pieces — **state preparation**, **QLSA**, and
    **readout** — are supplied as independent, swappable components::

        solver = QuantumLinearSolver(
            state_prep=DefaultStatePrep(),
            qlsa=HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle()),
            readout=MeasureXReadout(),
            backend=aer_backend,
        )
        result = solver.solve(A, b)   # returns SolveResult
        result.solution               # the post-processed solution vector
    """

    def __init__(
        self,
        qlsa: QLSA,
        readout: Readout,
        backend: Union[BackendV2, QuantinuumBackendConfig],
        *,
        state_prep: Optional[StatePrep] = None,   # defaults to DefaultStatePrep()
        shots: int = 1024,
        target_successful_shots: Optional[int] = None,
        shots_per_batch: Optional[int] = None,
        max_total_shots: Optional[int] = None,
        optimization_level: int = 3,
        ibm_options: Optional[IBMExecutionOptions] = None,
        executer: Optional[Executer] = None,
    ) -> None:
        self.qlsa = qlsa
        self.readout = readout
        self.state_prep = state_prep or DefaultStatePrep()
        self.backend = backend
        self.shots = shots
        self.target_successful_shots = target_successful_shots
        self.shots_per_batch = shots_per_batch or shots
        self.max_total_shots = max_total_shots
        self.optimization_level = optimization_level
        self.ibm_options = ibm_options
        self.executer = executer or Executer(ibm_options=ibm_options)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _is_quantinuum(self) -> bool:
        return isinstance(self.backend, QuantinuumBackendConfig)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        t0: Optional[float] = None,
        C: Optional[float] = None,
    ) -> SolveResult:
        """Run the full workflow and return a :class:`SolveResult`."""

        # 1. Build core QLSA circuit (no readout yet)
        qlsa_circuit: QLSACircuit = self.qlsa.build_circuit(
            A, b, self.state_prep, t0=t0, C=C,
        )

        # 2. Multi-circuit readouts (HRF, future shadow tomography, ...) own
        #    their own circuit-build → execute → combine flow.
        if isinstance(self.readout, MultiCircuitReadout):
            return self._solve_multi(qlsa_circuit, A, b, verbose=verbose)

        # 3. Single-circuit path: append readout measurements, transpile,
        #    execute, post-process.
        self.circuit = self.readout.apply(qlsa_circuit, state_prep=self.state_prep)

        transpiler = Transpiler(
            circuit=self.circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        self.transpiled_circuit = transpiler.optimize()

        if self.target_successful_shots is not None:
            return self._solve_until_successful_shots(
                self.transpiled_circuit, A, b, verbose=verbose,
                transpiler=transpiler,
                success_criterion=qlsa_circuit.success_criterion,
            )

        result = self._execute(transpiler, verbose=verbose)
        return _to_solve_result(self.readout.process(result, A, b, verbose=verbose))

    # ------------------------------------------------------------------
    # Multi-circuit solve path (HRF and any future shadow-tomography-style readout)
    # ------------------------------------------------------------------

    def _solve_multi(
        self,
        qlsa_circuit: QLSACircuit,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> SolveResult:
        """Run the build_circuits → execute-each → combine_results workflow.

        Generic over any :class:`~qlsas.readout.base.MultiCircuitReadout`
        implementation; no HRF-specific knowledge.
        """
        readout: MultiCircuitReadout = self.readout

        if self.target_successful_shots is not None:
            raise ValueError(
                f"target_successful_shots is not supported with "
                f"{type(readout).__name__}. Multi-circuit readouts require a "
                f"fixed shot budget across all circuits."
            )
        if self._is_quantinuum:
            raise NotImplementedError(
                f"{type(readout).__name__} does not yet support Quantinuum backends."
            )

        circuits = readout.build_circuits(qlsa_circuit)
        results: list[MeasurementResult] = []
        transpiled_circuits = []
        for circ in circuits:
            t = Transpiler(
                circuit=circ,
                backend=self.backend,
                optimization_level=self.optimization_level,
            )
            tc = t.optimize()
            transpiled_circuits.append(tc)
            self.transpiled_circuit = tc
            results.append(self._execute(t, verbose=verbose))

        # Restore the base circuit's transpiled form so self.transpiled_circuit
        # is sensible after solve() (refiner stashes it per iteration).
        self.transpiled_circuit = transpiled_circuits[0]

        tr = readout.combine_results(
            results, A, b,
            success_criterion=qlsa_circuit.success_criterion,
            verbose=verbose,
        )

        if verbose:
            print(
                f"{type(readout).__name__}: ran {len(circuits)} circuits, "
                f"avg success rate: {tr.success_rate:.3f}"
            )

        return SolveResult(
            solution=tr.scaled,
            direction=tr.direction,
            alpha=tr.alpha,
            success_rate=tr.success_rate,
            residual=tr.residual,
            metadata=dict(tr.metadata),
        )

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute(
        self, transpiler: Transpiler, verbose: bool = True, shots: int | None = None,
    ) -> MeasurementResult:
        effective_shots = shots if shots is not None else self.shots
        if self._is_quantinuum:
            return self.executer.run(
                self.transpiled_circuit,
                self.backend,
                effective_shots,
                verbose=verbose,
                register_infos=transpiler.register_infos,
                measurement_plan=transpiler.measurement_plan,
                optimization_level=self.optimization_level,
            )
        return self.executer.run(
            self.transpiled_circuit,
            self.backend,
            effective_shots,
            ibm_options=self.ibm_options,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Successful-shot accumulation
    # ------------------------------------------------------------------

    def _solve_until_successful_shots(
        self,
        transpiled_circuit,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        transpiler: Transpiler | None = None,
        success_criterion: SuccessCriterion | None = None,
    ) -> SolveResult:
        if self._is_quantinuum:
            return self._quantinuum_successful_shots(
                A, b, verbose=verbose, transpiler=transpiler,
                success_criterion=success_criterion,
            )
        return self._ibm_successful_shots(
            transpiled_circuit, A, b, verbose=verbose,
            success_criterion=success_criterion,
        )

    def _quantinuum_successful_shots(
        self,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        transpiler: Transpiler | None = None,
        success_criterion: SuccessCriterion | None = None,
    ) -> SolveResult:
        assert transpiler is not None
        total_shots = self.max_total_shots or self.shots
        result = self._execute(transpiler, verbose=verbose, shots=total_shots)
        counts = result.get_counts(self.readout.register_names)
        trimmed = _trim_counts_to_target(
            counts, self.target_successful_shots, success_criterion,
        )
        return _to_solve_result(
            self.readout.process(MeasurementResult(trimmed), A, b, verbose=verbose)
        )

    def _ibm_successful_shots(
        self,
        transpiled_circuit,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        success_criterion: SuccessCriterion | None = None,
    ) -> SolveResult:
        accumulated: dict[str, int] = defaultdict(int)
        num_successful_so_far = 0
        total_shots_so_far = 0
        total_shots_submitted = 0
        total_successful_seen = 0
        batch_size = self.shots_per_batch

        is_success = _success_predicate(success_criterion)

        opened_session = False
        if self.executer.session_active:
            self.executer.open_session(self.backend, verbose=verbose)
            opened_session = self.executer.session_active

        try:
            while True:
                result = self.executer.run(
                    transpiled_circuit,
                    self.backend,
                    batch_size,
                    ibm_options=self.ibm_options,
                    verbose=verbose,
                )
                counts = result.get_counts(self.readout.register_names)

                num_batch_successful = sum(
                    v for k, v in counts.items() if is_success(k)
                )
                total_shots_submitted += sum(counts.values())
                total_successful_seen += num_batch_successful

                if (
                    self.target_successful_shots is not None
                    and num_successful_so_far + num_batch_successful
                    <= self.target_successful_shots
                ):
                    for key, value in counts.items():
                        accumulated[key] += value
                    num_successful_so_far += num_batch_successful
                    total_shots_so_far += sum(counts.values())

                    if num_successful_so_far == self.target_successful_shots:
                        break
                else:
                    needed = (
                        self.target_successful_shots - num_successful_so_far
                        if self.target_successful_shots is not None
                        else 0
                    )
                    bitstrings = result.get_bitstrings(self.readout.register_names)

                    count_found = 0
                    cutoff = 0
                    for i, bs in enumerate(bitstrings):
                        if is_success(bs):
                            count_found += 1
                        if count_found == needed:
                            cutoff = i + 1
                            break

                    for bs in bitstrings[:cutoff]:
                        accumulated[bs] += 1

                    break

                if (
                    self.max_total_shots is not None
                    and total_shots_so_far >= self.max_total_shots
                ):
                    break
        finally:
            if opened_session:
                self.executer.close_session(verbose=verbose)

        final_successful = sum(v for k, v in accumulated.items() if is_success(k))
        hit_max_limit = (
            self.max_total_shots is not None
            and total_shots_so_far >= self.max_total_shots
        )
        if self.target_successful_shots is not None and not hit_max_limit:
            assert final_successful == self.target_successful_shots, (
                f"Expected exactly {self.target_successful_shots} successful shots, "
                f"but got {final_successful}."
            )

        success_probability = (
            total_successful_seen / total_shots_submitted
            if total_shots_submitted > 0
            else 0.0
        )

        # Keep legacy attributes for backward compatibility with Refiner.
        self.last_total_shots_submitted = total_shots_submitted
        self.last_total_successful_seen = total_successful_seen
        self.last_success_probability = success_probability

        proc = self.readout.process(
            MeasurementResult(dict(accumulated)), A, b, verbose=verbose
        )

        return _to_solve_result(
            proc,
            override_success_rate=success_probability,
            extra_metadata={
                "total_shots_submitted": total_shots_submitted,
                "total_successful_seen": total_successful_seen,
            },
        )


def _to_solve_result(
    proc_result,
    *,
    override_success_rate: float | None = None,
    extra_metadata: dict | None = None,
) -> SolveResult:
    """Wrap a readout's ``process()`` return value into a :class:`SolveResult`.

    Handles both :class:`TomographyResult` (tomography readouts) and the
    legacy ``(value, success_rate, residual)`` tuple still returned by
    :class:`~qlsas.readout.swap_test.SwapTestReadout`.
    """
    if isinstance(proc_result, TomographyResult):
        success_rate = (
            override_success_rate if override_success_rate is not None
            else proc_result.success_rate
        )
        metadata = dict(proc_result.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        return SolveResult(
            solution=proc_result.scaled,
            direction=proc_result.direction,
            alpha=proc_result.alpha,
            success_rate=success_rate,
            residual=proc_result.residual,
            metadata=metadata,
        )
    # Legacy 3-tuple path (SwapTestReadout): (value, success_rate, residual)
    value, success_rate, residual = proc_result
    if override_success_rate is not None:
        success_rate = override_success_rate
    return SolveResult(
        solution=value,
        success_rate=success_rate,
        residual=residual,
        metadata=dict(extra_metadata or {}),
    )


def _success_predicate(success_criterion: SuccessCriterion | None):
    """Return a ``str -> bool`` predicate for shot post-selection.

    Falls back to the legacy ``key[-1] == "1"`` rule when no criterion is
    supplied, preserving behaviour for synthetic tests.
    """
    if success_criterion is None:
        return lambda key: bool(key) and key[-1] == "1"
    return success_criterion.matches


def _trim_counts_to_target(
    counts: dict[str, int],
    target_successful: int | None,
    success_criterion: SuccessCriterion | None = None,
) -> dict[str, int]:
    """Keep all failed shots and exactly *target_successful* successful shots."""
    if target_successful is None:
        return counts

    is_success = _success_predicate(success_criterion)
    trimmed: dict[str, int] = {}
    found = 0
    for key, count in counts.items():
        if not is_success(key):
            trimmed[key] = count
            continue
        if found >= target_successful:
            continue
        take = min(count, target_successful - found)
        trimmed[key] = take
        found += take

    return trimmed
