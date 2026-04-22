"""Hadamard Random Forest readout for quantum state tomography.

Integrates the HRF algorithm (arXiv:2505.06455) into the QLSA readout framework.
HRF reduces the measurement circuit count from O(3^N) to O(N) by recovering
sign information from Hadamard-rotated measurements and majority-vote over
random spanning trees on the hypercube graph.

Only valid for real-valued solution states — HHL always produces real solutions
when A and b are real, which is the standard use case.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit

from qlsas.post_processor import norm_estimation
from qlsas.readout.base import Readout, QLSACircuit

if TYPE_CHECKING:
    from qlsas.measurement_result import MeasurementResult


class HRFReadout(Readout):
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
        # Populated by apply(); used by build_hrf_circuits() and _extract_probs()
        self._num_solution_qubits: int | None = None
        self._ancilla_creg_name: str | None = None
        self._solution_register = None
        self._ancilla_register = None
        self._base_circuit_core: QuantumCircuit | None = None

    # ------------------------------------------------------------------
    # Readout interface
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

    def apply(
        self,
        qlsa_circuit: QLSACircuit,
        *,
        state_prep=None,
    ) -> QuantumCircuit:
        """Append the solution-register measurement to the HHL core circuit.

        Also caches circuit metadata (register objects, ancilla classical register
        name) so that :meth:`build_hrf_circuits` can construct the N Hadamard
        variants without needing to reach back into the QLSA object.

        Parameters
        ----------
        qlsa_circuit : QLSACircuit
            Core HHL circuit (ancilla already measured mid-circuit).
        state_prep :
            Unused; accepted for interface compatibility.

        Returns
        -------
        QuantumCircuit
            Base circuit with solution measurement appended (no Hadamard gates).
        """
        self._num_solution_qubits = len(qlsa_circuit.solution_register)
        self._ancilla_creg_name = qlsa_circuit.ancilla_creg.name
        self._solution_register = qlsa_circuit.solution_register
        self._ancilla_register = qlsa_circuit.ancilla_register
        # Keep the pre-measurement core so build_hrf_circuits() can compose H variants
        self._base_circuit_core = qlsa_circuit.circuit.copy()

        circ = qlsa_circuit.circuit.copy()
        sol_creg = ClassicalRegister(
            self._num_solution_qubits, name=self._SOLUTION_CREG_NAME
        )
        circ.add_register(sol_creg)
        circ.measure(qlsa_circuit.solution_register, sol_creg)
        return circ

    def build_hrf_circuits(self) -> list[QuantumCircuit]:
        """Generate the N Hadamard-variant circuits (one per solution qubit).

        Each circuit is a copy of the full HHL core circuit with a Hadamard gate
        applied to one solution qubit before measurement, rotating its measurement
        basis from Z to X.  These circuits—together with the base circuit produced
        by :meth:`apply`—give the N+1 probability distributions that HRF needs.

        Returns
        -------
        list of QuantumCircuit
            N circuits, where the i-th circuit has H applied to ``solution_reg[i]``.
        """
        if self._base_circuit_core is None:
            raise RuntimeError(
                "apply() must be called before build_hrf_circuits()."
            )

        n = self._num_solution_qubits
        sol_reg = self._solution_register
        circuits: list[QuantumCircuit] = []

        for iq in range(n):
            circ = self._base_circuit_core.copy()
            sol_creg = ClassicalRegister(n, name=self._SOLUTION_CREG_NAME)
            circ.add_register(sol_creg)
            # Rotate qubit iq to the X basis before measuring
            circ.h(sol_reg[iq])
            circ.measure(sol_reg, sol_creg)
            circuits.append(circ)

        return circuits

    def process(
        self,
        result: list[np.ndarray],
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """Reconstruct the solution from post-selected probability distributions.

        Parameters
        ----------
        result : list of ndarray, shape (2^N,) each
            N+1 post-selected probability distributions over the solution register.
            ``result[0]`` is from the base circuit; ``result[k+1]`` is from the
            circuit with H on solution qubit k (0-indexed).
        A, b :
            Problem data for residual and norm-estimation.

        Returns
        -------
        solution : ndarray
            Reconstructed and physically-scaled solution vector.
        success_rate : float
            Placeholder (``nan``); the caller (:meth:`~qlsas.solver.QuantumLinearSolver._solve_hrf`)
            computes the actual ancilla success rate and stores it in :class:`~qlsas.solver.SolveResult`.
        residual : float
            ``‖A·solution − b‖``.
        """
        try:
            from hadamard_random_forest import get_statevector  # lazy import
        except ImportError as exc:
            raise ImportError(
                "HRFReadout requires the hadamard-random-forest package.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/comp-physics/Quantum-HRF-Tomography.git"
            ) from exc

        n = self._num_solution_qubits
        statevector = get_statevector(
            num_qubits=n,
            num_trees=self.num_trees,
            samples=result,
            save_tree=False,
            show_tree=False,
        )

        # HHL solution is real; discard floating-point imaginary residue
        solution = statevector.real
        norm = np.linalg.norm(solution)
        if norm < 1e-12:
            raise ValueError(
                "HRF reconstructed a near-zero solution vector. "
                "This usually means too few shots or an extremely low ancilla "
                "success rate. Try increasing shots."
            )
        solution = solution / norm

        alpha = norm_estimation(A, b, solution)
        solution_scaled = alpha * solution
        residual = float(np.linalg.norm(A @ solution_scaled - b))

        if verbose:
            print(f"HRF statevector norm (pre-scale): {norm:.4f}")
            print(f"HRF scale factor α:               {alpha:.4f}")
            print(f"solver residual:                  {residual:.6f}")

        return solution_scaled, float("nan"), residual

    # ------------------------------------------------------------------
    # Internal helper called by solver._solve_hrf
    # ------------------------------------------------------------------

    def _extract_probs(
        self,
        result: "MeasurementResult",
        n_sol: int,
    ) -> tuple[np.ndarray, float]:
        """Extract ancilla-postselected probabilities from one circuit's result.

        Filters measurement counts to shots where the ancilla flag is 1 (success)
        and returns the marginal probability distribution over the solution register
        together with the raw ancilla success rate.

        Parameters
        ----------
        result : MeasurementResult
            Wrapped measurement counts from the executer.
        n_sol : int
            Number of solution qubits (= log2(len(b))).

        Returns
        -------
        probs : ndarray, shape (2^n_sol,)
            Normalised probability distribution over the solution register,
            conditioned on ancilla = 1.
        success_rate : float
            Fraction of total shots that had ancilla = 1.
        """
        counts = result.get_counts(self.register_names)
        probs = np.zeros(2**n_sol, dtype=float)
        total_good = 0
        total_shots = sum(counts.values())

        for key, count in counts.items():
            # Bitstring format: [n_sol solution bits][1 ancilla bit]
            # The first name in register_names (ancilla) appears at the right.
            if key[-1] == "1":
                sol_bits = key[:n_sol]
                idx = int(sol_bits, 2)
                probs[idx] += count
                total_good += count

        if total_good == 0:
            raise ValueError(
                "No successful ancilla shots found in HRF circuit. "
                "Increase shots or check the circuit."
            )

        success_rate = total_good / total_shots if total_shots > 0 else 0.0
        return probs / total_good, success_rate
