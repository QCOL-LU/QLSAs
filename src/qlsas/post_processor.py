"""Pure-computation post-processing for quantum linear solver results.

:class:`Post_Processor` operates exclusively on plain ``dict[str, int]``
counts mappings.  All backend-specific dispatch (e.g. converting a Qiskit
``SamplerPubResult``) is handled upstream by the
:class:`~qlsas.readout.base.Readout` strategy that owns the measurement
registers.

Module-level function
---------------------
:func:`norm_estimation`
    Compute the scalar α such that α·x best fits Ax = b.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.linalg as LA


def norm_estimation(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Estimate the scalar α such that α·x best approximates the true solution.

    Minimises ||A·(α·x) − b||² by solving the 1-D least-squares problem,
    which gives α = (A·x)ᵀ b / ||A·x||².
    """
    v = A @ x
    denominator = np.dot(v, v)
    if denominator == 0:
        return 1e-10
    return np.dot(v, b) / denominator


class Post_Processor:
    """Post-processing for quantum linear solver results.

    All public methods accept plain ``dict[str, int]`` counts mappings.
    Readout strategies are responsible for extracting counts from raw
    backend results before calling these methods.
    """

    # ------------------------------------------------------------------
    # norm_estimation — convenience delegate
    # ------------------------------------------------------------------

    def norm_estimation(self, A, b, x):
        """Delegate to the module-level :func:`norm_estimation`."""
        return norm_estimation(A, b, x)

    # ------------------------------------------------------------------
    # Tomography
    # ------------------------------------------------------------------

    def tomography_from_counts(
        self,
        counts: dict[str, int],
        A: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Reconstruct the solution vector from a counts dict.

        Bitstring convention: ``x_result`` bits + ancilla flag as the last bit;
        success shots have last bit ``'1'``.

        Returns ``(solution, success_rate, residual)``.
        """
        x_size = int(math.log2(len(b)))
        num_successful_shots = 0
        approximate_solution = np.zeros(len(b))
        total_shots = sum(counts.values())

        for key, value in counts.items():
            if key[-1] == "1":
                num_successful_shots += value
                coord = int(key[:x_size], base=2)
                approximate_solution[coord] = value

        if num_successful_shots == 0:
            raise ValueError("No successful shots.")

        approximate_solution = np.sqrt(approximate_solution / num_successful_shots)
        success_rate = num_successful_shots / total_shots if total_shots else 0.0
        return self._finish_tomography(
            approximate_solution, success_rate, num_successful_shots, total_shots, A, b
        )

    # ------------------------------------------------------------------
    # Swap test
    # ------------------------------------------------------------------

    def swap_test_from_counts(
        self,
        counts: dict[str, int],
        A: np.ndarray,
        b: np.ndarray,
        swap_test_vector: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute the swap-test expected value from a counts dict.

        Bitstring convention: swap-test ancilla bit first, HHL ancilla flag last;
        success shots have last bit ``'1'``.

        Returns ``(expected_value, success_rate, residual)``.
        """
        correct_shots = 0
        num_swap_ones = 0
        total_shots = sum(counts.values())

        for key, value in counts.items():
            if key[-1] == "1":
                correct_shots += value
                if key[0] == "1":
                    num_swap_ones += value

        if total_shots == 0:
            raise ValueError("No shots recorded.")
        if correct_shots == 0:
            raise ValueError("No successful HHL shots.")

        success_rate = correct_shots / total_shots
        exp_value = num_swap_ones / correct_shots

        classical_solution = LA.solve(A, b)
        normalized_classical = (
            classical_solution / LA.norm(classical_solution)
            if LA.norm(classical_solution) > 0
            else classical_solution
        )
        normalized_swap = (
            swap_test_vector / LA.norm(swap_test_vector)
            if LA.norm(swap_test_vector) > 0
            else swap_test_vector
        )

        overlap = np.vdot(normalized_swap, normalized_classical)
        expected_prob = 0.5 - 0.5 * (np.abs(overlap) ** 2)
        residual = abs(exp_value - expected_prob)
        return exp_value, success_rate, residual

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finish_tomography(
        self,
        approximate_solution: np.ndarray,
        success_rate: float,
        num_successful_shots: int,
        total_shots: int,
        A: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Apply sign correction, unit-norm check, and compute residual."""
        classical_solution = LA.solve(A, b)
        for i in range(len(approximate_solution)):
            approximate_solution[i] *= np.sign(classical_solution[i])

        assert np.allclose(
            sum(approximate_solution[i] ** 2 for i in range(len(approximate_solution))),
            1.0,
            atol=1e-6,
        ), "Approximate solution is not normalized."

        scaling_factor = norm_estimation(A, b, approximate_solution)
        scaled_solution = approximate_solution * scaling_factor
        residual = np.linalg.norm(b - A @ scaled_solution)
        return approximate_solution, success_rate, residual
