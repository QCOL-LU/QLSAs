"""Pure-computation post-processing for quantum linear solver results.

This module exposes free functions that operate on plain ``dict[str, int]``
counts mappings. All backend-specific dispatch (e.g. converting a Qiskit
``SamplerPubResult``) is handled upstream by the
:class:`~qlsas.readout.base.Readout` strategy that owns the measurement
registers.

Functions
---------
:func:`norm_estimation`
    Compute the scalar α such that α·x best fits Ax = b.
:func:`tomography_from_counts`
    Reconstruct a unit-norm solution vector from a counts dict.
:func:`swap_test_from_counts`
    Compute the swap-test expected value from a counts dict.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as LA

if TYPE_CHECKING:
    from qlsas.readout.base import SuccessCriterion, TomographyResult


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


def _is_success(key: str, success_criterion: "SuccessCriterion | None") -> bool:
    """Check whether a bitstring marks a successful shot.

    Falls back to the legacy HHL convention (``key[-1] == "1"``) when no
    criterion is supplied — used by tests with synthetic counts.
    """
    if success_criterion is None:
        return bool(key) and key[-1] == "1"
    return success_criterion.matches(key)


def tomography_from_counts(
    counts: dict[str, int],
    A: np.ndarray,
    b: np.ndarray,
    success_criterion: "SuccessCriterion | None" = None,
) -> "TomographyResult":
    """Reconstruct the solution vector from a counts dict.

    Bitstring convention: solution-register bits occupy the leftmost
    ``log2(len(b))`` characters; success-ancilla bits occupy the rightmost
    characters as defined by *success_criterion* (or, when omitted,
    the legacy single-ancilla ``key[-1] == "1"`` rule).

    Returns a :class:`~qlsas.readout.base.TomographyResult`. The object is
    iterable as ``(direction, success_rate, residual)`` so existing tuple
    unpacking continues to work.
    """
    x_size = int(math.log2(len(b)))
    num_successful_shots = 0
    approximate_solution = np.zeros(len(b))
    total_shots = sum(counts.values())

    for key, value in counts.items():
        if _is_success(key, success_criterion):
            num_successful_shots += value
            coord = int(key[:x_size], base=2)
            approximate_solution[coord] = value

    if num_successful_shots == 0:
        raise ValueError("No successful shots.")

    approximate_solution = np.sqrt(approximate_solution / num_successful_shots)
    success_rate = num_successful_shots / total_shots if total_shots else 0.0
    return _finish_tomography(
        approximate_solution, success_rate, num_successful_shots, total_shots, A, b
    )


def swap_test_from_counts(
    counts: dict[str, int],
    A: np.ndarray,
    b: np.ndarray,
    swap_test_vector: np.ndarray,
    success_criterion: "SuccessCriterion | None" = None,
) -> tuple[float, float, float]:
    """Compute the swap-test expected value from a counts dict.

    Bitstring convention: swap-test ancilla bit is the leftmost character;
    success-ancilla bits are the rightmost characters as defined by
    *success_criterion* (default: single-ancilla ``key[-1] == "1"``).

    Returns ``(expected_value, success_rate, residual)``.
    """
    correct_shots = 0
    num_swap_ones = 0
    total_shots = sum(counts.values())

    for key, value in counts.items():
        if _is_success(key, success_criterion):
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


def _finish_tomography(
    approximate_solution: np.ndarray,
    success_rate: float,
    num_successful_shots: int,
    total_shots: int,
    A: np.ndarray,
    b: np.ndarray,
) -> "TomographyResult":
    """Apply sign correction, unit-norm check, and compute residual.

    Returns a :class:`~qlsas.readout.base.TomographyResult` carrying the
    unit-norm direction, the least-squares scale α, the success rate, and
    the residual ``‖b − A·(α·direction)‖``.
    """
    from qlsas.readout.base import TomographyResult

    classical_solution = LA.solve(A, b)
    for i in range(len(approximate_solution)):
        approximate_solution[i] *= np.sign(classical_solution[i])

    assert np.allclose(
        sum(approximate_solution[i] ** 2 for i in range(len(approximate_solution))),
        1.0,
        atol=1e-6,
    ), "Approximate solution is not normalized."

    alpha = norm_estimation(A, b, approximate_solution)
    residual = float(np.linalg.norm(b - A @ (alpha * approximate_solution)))
    return TomographyResult(
        direction=approximate_solution,
        alpha=float(alpha),
        success_rate=success_rate,
        residual=residual,
    )
