"""
Shared functionality for HHL-style solvers across different backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from numpy import linalg as LA

from .solver import Context, Solver, SolverResult


@dataclass
class CountsSummary:
    amplitudes: np.ndarray
    success_counts: int
    total_counts: int


class HHLBaseSolver(Solver):
    """
    Partial implementation of the Solver interface tailored to HHL workflows.

    Subclasses typically only need to implement `_build_circuit` and
    `_execute_circuit`, then call `_build_solver_result` during post-processing.
    """

    def __init__(
        self,
        default_n_qpe_qubits: int = 3,
        default_t0: float = 2 * np.pi,
        enforce_hermitian: bool = True,
    ) -> None:
        self.default_n_qpe_qubits = default_n_qpe_qubits
        self.default_t0 = default_t0
        self.enforce_hermitian = enforce_hermitian

    def _prepare_problem(self, A: np.ndarray, b: np.ndarray, **options: Any) -> Context:
        context = super()._prepare_problem(A, b, **options)
        A_arr = context["A"]
        b_arr = context["b"]

        if self.enforce_hermitian and not np.allclose(
            A_arr, A_arr.conjugate().T, atol=1e-8
        ):
            raise ValueError("Matrix A must be Hermitian for HHL-based solvers.")

        vector_length = b_arr.shape[0]
        if vector_length == 0 or (vector_length & (vector_length - 1)) != 0:
            raise ValueError("Length of b must be a non-zero power of two for HHL.")

        n_state_qubits = int(np.log2(vector_length))

        norm_b = LA.norm(b_arr)
        if np.isclose(norm_b, 0.0):
            raise ValueError("Vector b must have non-zero norm.")

        scaled_A = A_arr / norm_b
        scaled_b = b_arr / norm_b

        try:
            classical_solution = LA.solve(A_arr, b_arr)
            classical_solution_scaled = LA.solve(scaled_A, scaled_b)
        except LA.LinAlgError as exc:
            raise ValueError("Matrix A must be invertible for HHL.") from exc

        n_qpe_qubits = options.get("n_qpe_qubits", self.default_n_qpe_qubits)
        if not isinstance(n_qpe_qubits, int) or n_qpe_qubits <= 0:
            raise ValueError("n_qpe_qubits must be a positive integer.")

        t0 = options.get("t0", self.default_t0)

        context.update(
            {
                "n_state_qubits": n_state_qubits,
                "norm_b": norm_b,
                "scaled_A": scaled_A,
                "scaled_b": scaled_b,
                "classical_solution": classical_solution,
                "classical_solution_scaled": classical_solution_scaled,
                "n_qpe_qubits": n_qpe_qubits,
                "t0": t0,
            }
        )
        return context

    def _build_solver_result(
        self,
        counts_summary: CountsSummary,
        execution_metadata: Dict[str, Any],
        context: Context,
        raw_result: Any = None,
    ) -> SolverResult:
        """
        Convert aggregated counts into a SolverResult while computing metrics.
        """

        classical_scaled = context["classical_solution_scaled"]
        adjusted = counts_summary.amplitudes.astype(float).copy()

        for idx, reference_value in enumerate(classical_scaled):
            if np.isclose(reference_value, 0.0):
                continue
            if np.real(reference_value) < 0:
                adjusted[idx] = -abs(adjusted[idx])

        two_norm_error = LA.norm(classical_scaled - adjusted)
        residual_error = LA.norm(context["b"] - context["A"] @ adjusted)
        success_probability = (
            counts_summary.success_counts / counts_summary.total_counts
            if counts_summary.total_counts
            else 0.0
        )

        metadata: Dict[str, Any] = {
            "two_norm_error": two_norm_error,
            "residual_error": residual_error,
            "success_counts": counts_summary.success_counts,
            "total_counts": counts_summary.total_counts,
            "success_probability": success_probability,
            "n_qpe_qubits": context["n_qpe_qubits"],
            "norm_b": context["norm_b"],
        }
        metadata.update(execution_metadata)

        return SolverResult(
            solution=adjusted,
            metadata=metadata,
            raw_result=raw_result,
        )

    def _counts_to_amplitudes(
        self, counts: Dict[Any, int], n_state_qubits: int
    ) -> CountsSummary:
        """
        Convert measurement counts into amplitude estimates for the solution register.
        """

        total_bits = n_state_qubits + 1  # solution register + success ancilla
        amplitudes = np.zeros(2**n_state_qubits, dtype=float)
        success_counts = 0
        total_counts = 0

        for key, value in counts.items():
            shots = int(value)
            total_counts += shots

            bits = self._normalise_count_key(key, total_bits)
            if not bits or len(bits) < total_bits:
                continue

            anc_bit = bits[-1]
            if anc_bit != "1":
                continue

            register_bits = bits[:n_state_qubits]
            try:
                index = int(register_bits, 2)
            except ValueError:
                continue

            amplitudes[index] += shots
            success_counts += shots

        if success_counts > 0:
            amplitudes = np.sqrt(amplitudes / success_counts)

        return CountsSummary(amplitudes=amplitudes, success_counts=success_counts, total_counts=total_counts)

    @staticmethod
    def _normalise_count_key(key: Any, total_bits: int) -> str:
        """
        Convert a backend-specific count key into a binary string.
        """

        if isinstance(key, (tuple, list)):
            bits = "".join(str(bit) for bit in key)
        elif isinstance(key, int):
            bits = format(key, f"0{total_bits}b")
        else:
            bits = str(key).replace(" ", "")

        if len(bits) < total_bits:
            bits = bits.zfill(total_bits)
        elif len(bits) > total_bits:
            bits = bits[-total_bits:]
        return bits


