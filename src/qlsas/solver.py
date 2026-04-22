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
from qlsas.readout.base import Readout, QLSACircuit
from qlsas.measurement_result import MeasurementResult
from qlsas.transpiler import Transpiler


@dataclass
class SolveResult:
    """Value object returned by :meth:`QuantumLinearSolver.solve`.

    Attributes:
        solution: Post-processed solution vector.
        success_rate: Fraction of shots that passed the ancilla post-selection
            (``None`` when batching is not used).
        residual: Norm of the residual ``‖Ax − b‖`` (``None`` when the caller
            does not supply *A* and *b* at construction time).
        metadata: Any additional algorithm-specific diagnostics.
    """

    solution: np.ndarray
    success_rate: Optional[float] = None
    residual: Optional[float] = None
    metadata: dict = field(default_factory=dict)

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
            qlsa=HHL(num_qpe_qubits=4, eig_oracle=ClassicalEigOracle()),
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

        # 2. Append readout measurements, passing state_prep for strategies
        #    that need it (e.g. SwapTestReadout).
        self.circuit = self.readout.apply(qlsa_circuit, state_prep=self.state_prep)

        # 3. Transpile
        transpiler = Transpiler(
            circuit=self.circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        self.transpiled_circuit = transpiler.optimize()

        # 4. Execute + post-process
        from qlsas.readout.hrf_readout import HRFReadout
        if isinstance(self.readout, HRFReadout):
            return self._solve_hrf(qlsa_circuit, transpiler, A, b, verbose=verbose)

        if self.target_successful_shots is not None:
            return self._solve_until_successful_shots(
                self.transpiled_circuit, A, b, verbose=verbose,
                transpiler=transpiler,
            )

        result = self._execute(transpiler, verbose=verbose)
        solution = self.readout.process(result, A, b, verbose=verbose)[0]
        return SolveResult(solution=solution)

    # ------------------------------------------------------------------
    # HRF multi-circuit solve path
    # ------------------------------------------------------------------

    def _solve_hrf(
        self,
        qlsa_circuit: QLSACircuit,
        base_transpiler: "Transpiler",
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> "SolveResult":
        """Run the N+1-circuit HRF tomography workflow.

        Executes the base measurement circuit followed by one Hadamard-variant
        circuit per solution qubit, post-selects on ancilla success, and
        reconstructs the solution via majority-vote sign recovery.

        Parameters
        ----------
        qlsa_circuit : QLSACircuit
            Core HHL circuit (output of ``qlsa.build_circuit``).
        base_transpiler : Transpiler
            Pre-built transpiler whose ``.optimize()`` result is already stored
            in ``self.transpiled_circuit``.
        """
        from qlsas.readout.hrf_readout import HRFReadout
        readout: HRFReadout = self.readout

        if self.target_successful_shots is not None:
            raise ValueError(
                "target_successful_shots is not supported with HRFReadout. "
                "HRF requires a fixed shot budget across all N+1 circuits."
            )
        if self._is_quantinuum:
            raise NotImplementedError(
                "HRFReadout does not yet support Quantinuum backends."
            )

        n_sol = len(qlsa_circuit.solution_register)
        base_transpiled = self.transpiled_circuit  # set by solve() before this call

        # --- Build and transpile the N Hadamard-variant circuits ---
        h_circuits = readout.build_hrf_circuits()
        h_transpilers: list[Transpiler] = []
        h_transpiled = []
        for hc in h_circuits:
            t = Transpiler(
                circuit=hc,
                backend=self.backend,
                optimization_level=self.optimization_level,
            )
            h_transpilers.append(t)
            h_transpiled.append(t.optimize())

        # --- Execute base circuit ---
        self.transpiled_circuit = base_transpiled
        base_result = self._execute(base_transpiler, verbose=verbose)
        base_probs, base_rate = readout._extract_probs(base_result, n_sol)

        # --- Execute each Hadamard-variant circuit ---
        h_samples: list[np.ndarray] = []
        h_rates: list[float] = []
        for t, tc in zip(h_transpilers, h_transpiled):
            self.transpiled_circuit = tc
            result = self._execute(t, verbose=verbose)
            probs, rate = readout._extract_probs(result, n_sol)
            h_samples.append(probs)
            h_rates.append(rate)

        # Restore base so self.transpiled_circuit is sensible after solve()
        self.transpiled_circuit = base_transpiled

        all_samples = [base_probs] + h_samples
        avg_success_rate = float(np.mean([base_rate] + h_rates))

        solution_scaled, _, residual = readout.process(
            all_samples, A, b, verbose=verbose
        )

        if verbose:
            print(
                f"HRF: ran {n_sol + 1} circuits "
                f"(1 base + {n_sol} Hadamard variants), "
                f"avg ancilla success rate: {avg_success_rate:.3f}"
            )

        return SolveResult(
            solution=solution_scaled,
            success_rate=avg_success_rate,
            residual=residual,
            metadata={"num_hrf_circuits": n_sol + 1, "num_trees": readout.num_trees},
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
    ) -> SolveResult:
        if self._is_quantinuum:
            return self._quantinuum_successful_shots(
                A, b, verbose=verbose, transpiler=transpiler,
            )
        return self._ibm_successful_shots(
            transpiled_circuit, A, b, verbose=verbose,
        )

    def _quantinuum_successful_shots(
        self,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        transpiler: Transpiler | None = None,
    ) -> SolveResult:
        assert transpiler is not None
        total_shots = self.max_total_shots or self.shots
        result = self._execute(transpiler, verbose=verbose, shots=total_shots)
        counts = result.get_counts(self.readout.register_names)
        trimmed = _trim_counts_to_target(counts, self.target_successful_shots)
        solution = self.readout.process(MeasurementResult(trimmed), A, b, verbose=verbose)[0]
        return SolveResult(solution=solution)

    def _ibm_successful_shots(
        self,
        transpiled_circuit,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> SolveResult:
        accumulated: dict[str, int] = defaultdict(int)
        num_successful_so_far = 0
        total_shots_so_far = 0
        total_shots_submitted = 0
        total_successful_seen = 0
        batch_size = self.shots_per_batch

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
                    v for k, v in counts.items() if k[-1] == "1"
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
                        if bs[-1] == "1":
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

        final_successful = sum(v for k, v in accumulated.items() if k[-1] == "1")
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

        solution = self.readout.process(
            MeasurementResult(dict(accumulated)), A, b, verbose=verbose
        )[0]

        return SolveResult(
            solution=solution,
            success_rate=success_probability,
            metadata={
                "total_shots_submitted": total_shots_submitted,
                "total_successful_seen": total_successful_seen,
            },
        )


def _trim_counts_to_target(
    counts: dict[str, int],
    target_successful: int | None,
) -> dict[str, int]:
    """Keep all failed shots and exactly *target_successful* successful shots."""
    if target_successful is None:
        return counts

    trimmed: dict[str, int] = {}
    found = 0
    for key, count in counts.items():
        if key[-1] != "1":
            trimmed[key] = count
            continue
        if found >= target_successful:
            continue
        take = min(count, target_successful - found)
        trimmed[key] = take
        found += take

    return trimmed
