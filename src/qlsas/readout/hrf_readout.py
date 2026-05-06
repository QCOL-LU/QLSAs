"""Hadamard Random Forest readout for quantum state tomography.

Integrates the HRF algorithm (arXiv:2505.06455) into the QLSA readout framework.
HRF reduces the measurement circuit count from O(3^N) to O(N) by recovering
sign information from Hadamard-rotated measurements and majority-vote over
random spanning trees on the hypercube graph.

Only valid for real-valued solution states — HHL always produces real solutions
when A and b are real, which is the standard use case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit

from qlsas.post_processor import norm_estimation
from qlsas.readout.base import (
    MultiCircuitReadout,
    QLSACircuit,
    SuccessCriterion,
    TomographyResult,
)

if TYPE_CHECKING:
    from qlsas.measurement_result import MeasurementResult


class HRFReadout(MultiCircuitReadout):
    """Reconstruct the HHL solution via Hadamard Random Forest (HRF) tomography.

    Standard tomography requires O(3^N) circuits to reconstruct an N-qubit state.
    HRF reduces this to N+1 circuits: one base measurement and one Hadamard-rotated
    variant per solution qubit.  Signs are recovered via majority voting over
    random spanning trees on the N-dimensional hypercube graph.

    Parameters
    ----------
    num_trees : int
        Number of random spanning trees used for sign reconstruction via
        majority vote.  More trees improve robustness at the cost of classical
        post-processing time.  10–50 is typically sufficient for small systems;
        the paper uses O(N) trees for N-qubit states.

    Notes
    -----
    * HRF assumes **real-valued** quantum state amplitudes.  This holds for HHL
      when A is real symmetric and b is real, which covers the common case.
    * The ``target_successful_shots`` option in :class:`~qlsas.solver.QuantumLinearSolver`
      is not compatible with HRFReadout (HRF requires a fixed shot budget per circuit).
    * ``qiskit-experiments`` is an optional dependency of the HRF package that is
      not installed by ``qlsas``; the core tomography path does not require it.
    """

    _SOLUTION_CREG_NAME: str = "hrf_x_result"

    def __init__(self, num_trees: int = 20) -> None:
        self.num_trees = num_trees
        # Populated by build_circuits(); consumed by register_names and combine_results.
        self._num_solution_qubits: int | None = None
        self._ancilla_creg_name: str | None = None
        self._success_criterion: SuccessCriterion | None = None

    # ------------------------------------------------------------------
    # MultiCircuitReadout interface
    # ------------------------------------------------------------------

    @property
    def register_names(self) -> list[str]:
        """Classical register names in join_data order.

        Per the Qiskit join_data convention used throughout this codebase,
        the *first* name's bits end up as the rightmost (LSB) characters of
        each combined bitstring.  Concretely:

        * ``ancilla_flag_result`` → last character  (``key[-1]``)
        * ``hrf_x_result``        → first N characters (``key[:N]``)
        """
        return [
            self._ancilla_creg_name or "ancilla_flag_result",
            self._SOLUTION_CREG_NAME,
        ]

    def build_circuits(self, qlsa_circuit: QLSACircuit) -> list[QuantumCircuit]:
        """Return the base circuit + N Hadamard-variant circuits.

        Caches the bits of *qlsa_circuit* metadata that
        :meth:`register_names` and :meth:`combine_results` need later
        (the ancilla creg name, the success criterion, the solution-register
        width). Otherwise stateless: callers may invoke this multiple times.
        """
        n_sol = len(qlsa_circuit.solution_register)
        sol_reg = qlsa_circuit.solution_register
        self._ancilla_creg_name = qlsa_circuit.ancilla_creg.name
        self._success_criterion = qlsa_circuit.success_criterion
        self._num_solution_qubits = n_sol

        circuits: list[QuantumCircuit] = []

        base = qlsa_circuit.circuit.copy()
        base_creg = ClassicalRegister(n_sol, name=self._SOLUTION_CREG_NAME)
        base.add_register(base_creg)
        base.measure(sol_reg, base_creg)
        circuits.append(base)

        for iq in range(n_sol):
            circ = qlsa_circuit.circuit.copy()
            sol_creg = ClassicalRegister(n_sol, name=self._SOLUTION_CREG_NAME)
            circ.add_register(sol_creg)
            # Rotate qubit iq to the X basis before measuring.
            circ.h(sol_reg[iq])
            circ.measure(sol_reg, sol_creg)
            circuits.append(circ)

        return circuits

    def combine_results(
        self,
        results: list["MeasurementResult"],
        A: np.ndarray,
        b: np.ndarray,
        success_criterion: SuccessCriterion | None = None,
        verbose: bool = True,
    ) -> TomographyResult:
        """Reconstruct a :class:`TomographyResult` from N+1 per-circuit results.

        Each entry of *results* is the raw measurement result for one circuit.
        The first entry is the base circuit; the rest are the Hadamard variants
        in qubit order. *success_criterion* overrides the criterion cached
        during :meth:`build_circuits` if supplied.
        """
        try:
            from hadamard_random_forest import get_statevector  # lazy import
        except ImportError as exc:
            raise ImportError(
                "HRFReadout requires the hadamard-random-forest package.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/comp-physics/Quantum-HRF-Tomography.git"
            ) from exc

        criterion = (
            success_criterion if success_criterion is not None
            else self._success_criterion
        )
        n_sol = self._num_solution_qubits
        if n_sol is None:
            raise RuntimeError(
                "HRFReadout.combine_results() called before build_circuits(). "
                "Use QuantumLinearSolver to drive the multi-circuit workflow."
            )

        all_probs: list[np.ndarray] = []
        all_rates: list[float] = []
        for r in results:
            probs, rate = self._postselect_probs(r, n_sol, criterion)
            all_probs.append(probs)
            all_rates.append(rate)

        statevector = get_statevector(
            num_qubits=n_sol,
            num_trees=self.num_trees,
            samples=all_probs,
            save_tree=False,
            show_tree=False,
        )

        # HHL solution is real; discard floating-point imaginary residue.
        direction = statevector.real
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError(
                "HRF reconstructed a near-zero solution vector. "
                "This usually means too few shots or an extremely low ancilla "
                "success rate. Try increasing shots."
            )
        direction = direction / norm

        alpha = float(norm_estimation(A, b, direction))
        residual = float(np.linalg.norm(A @ (alpha * direction) - b))
        avg_success_rate = float(np.mean(all_rates))

        if verbose:
            print(f"HRF statevector norm (pre-scale): {norm:.4f}")
            print(f"HRF scale factor α:               {alpha:.4f}")
            print(f"avg ancilla success rate:         {avg_success_rate:.4f}")
            print(f"solver residual:                  {residual:.6f}")

        return TomographyResult(
            direction=direction,
            alpha=alpha,
            success_rate=avg_success_rate,
            residual=residual,
            metadata={
                "num_hrf_circuits": len(results),
                "num_trees": self.num_trees,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _postselect_probs(
        self,
        result: "MeasurementResult",
        n_sol: int,
        success_criterion: SuccessCriterion | None,
    ) -> tuple[np.ndarray, float]:
        """Filter to successful shots and return the marginal solution-register
        probability distribution along with the ancilla success rate.

        Bitstring layout: solution-register bits are the leftmost ``n_sol``
        characters; success-criterion bits are the rightmost characters
        (per the Qiskit join_data convention used by :meth:`register_names`).
        """
        filtered, total_good, total_shots = result.get_postselected_counts(
            self.register_names, success_criterion,
        )
        probs = np.zeros(2**n_sol, dtype=float)
        for key, count in filtered.items():
            probs[int(key[:n_sol], 2)] += count
        if total_good == 0:
            raise ValueError(
                "No successful ancilla shots found in HRF circuit. "
                "Increase shots or check the circuit."
            )
        rate = total_good / total_shots if total_shots > 0 else 0.0
        return probs / total_good, rate
