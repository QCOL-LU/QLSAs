"""Tests for qlsas.post_processor module-level functions."""

import numpy as np
import numpy.linalg as LA
import pytest

from qlsas.post_processor import (
    norm_estimation,
    tomography_from_counts,
    _finish_tomography,
)
from qlsas.state_prep import StatePrep, DefaultStatePrep
from qlsas.algorithms.hhl import HHL, MCRYEigOracle
from qlsas.readout import MeasureXReadout, SwapTestReadout
from qlsas.transpiler import Transpiler
from qlsas.executer import Executer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized(v):
    return v / LA.norm(v)


def _run_hhl_2x2_measure_x(aer_backend, A, b, shots=4096):
    """Build, transpile, and execute a 2x2 measure_x HHL circuit; return the raw result."""
    sp = DefaultStatePrep()
    hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
    qlsa_circuit = hhl.build_circuit(A, b, sp)
    readout = MeasureXReadout()
    circ = readout.apply(qlsa_circuit)
    transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=1)
    tc = transpiler.optimize()
    ex = Executer()
    return ex.run(tc, aer_backend, shots=shots, verbose=False)


def _run_hhl_2x2_swap_test(aer_backend, A, b, swap_vec, shots=4096):
    """Build, transpile, and execute a 2x2 swap_test HHL circuit; return the raw result."""
    sp = DefaultStatePrep()
    hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
    qlsa_circuit = hhl.build_circuit(A, b, sp)
    readout = SwapTestReadout(swap_test_vector=swap_vec, state_prep=sp)
    circ = readout.apply(qlsa_circuit)
    transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=1)
    tc = transpiler.optimize()
    ex = Executer()
    return ex.run(tc, aer_backend, shots=shots, verbose=False)


# ===================================================================
# norm_estimation
# ===================================================================

class TestNormEstimation:

    def test_known_system(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x_true = LA.solve(A, b)
        x_norm = _normalized(x_true)
        alpha = norm_estimation(A, b, x_norm)
        scaled = alpha * x_norm
        assert np.allclose(A @ scaled, b, atol=1e-6)

    def test_zero_denominator(self):
        A = np.array([[0.0, 0.0], [0.0, 0.0]])
        b = np.array([1.0, 0.0])
        x = np.array([0.0, 0.0])
        result = norm_estimation(A, b, x)
        assert result == pytest.approx(1e-10)


# ===================================================================
# tomography_from_counts
# ===================================================================

class TestTomographyFromCounts:

    def test_synthetic_counts_2x2(self):
        """For a 2x2 system, x_size=1. Construct counts where coord 0 gets 75 hits, coord 1 gets 25."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = _normalized(np.array([1.0, 1.0]))

        counts = {
            "01": 75,   # coord 0, ancilla=1 (success)
            "11": 25,   # coord 1, ancilla=1 (success)
            "00": 50,   # coord 0, ancilla=0 (fail)
            "10": 50,   # coord 1, ancilla=0 (fail)
        }
        solution, success_rate, residual = tomography_from_counts(counts, A, b)

        assert len(solution) == 2
        assert success_rate == pytest.approx(100 / 200)
        assert np.isfinite(residual)

    def test_no_successful_shots_raises(self):
        A = np.eye(2)
        b = _normalized(np.array([1.0, 0.0]))
        counts = {"00": 50, "10": 50}
        with pytest.raises(ValueError, match="No successful shots"):
            tomography_from_counts(counts, A, b)

    def test_solution_is_unit_normalized(self):
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        b = _normalized(np.array([1.0, 1.0]))
        counts = {"01": 60, "11": 40, "00": 100}
        solution, _, _ = tomography_from_counts(counts, A, b)
        assert np.isclose(LA.norm(solution), 1.0, atol=1e-5)


# ===================================================================
# _finish_tomography — sign correction
# ===================================================================

class TestFinishTomography:

    def test_signs_match_classical_solution(self):
        A = np.array([[2.0, -1.0], [-1.0, 2.0]])
        b = _normalized(np.array([1.0, -1.0]))
        classical = LA.solve(A, b)

        approx_sol = np.abs(_normalized(classical))
        solution, _, _ = _finish_tomography(
            approx_sol.copy(), 0.5, 100, 200, A, b
        )
        for i in range(len(solution)):
            assert np.sign(solution[i]) == np.sign(classical[i]) or np.isclose(solution[i], 0, atol=1e-10)


# ===================================================================
# Integration: process via MeasureXReadout
# ===================================================================

class TestProcessQiskitTomography:

    def test_solution_direction_close_to_classical(self, aer_backend, pd_2x2, b_2):
        result = _run_hhl_2x2_measure_x(aer_backend, pd_2x2, b_2, shots=4096)
        readout = MeasureXReadout()
        solution, success_rate, residual = readout.process(result, pd_2x2, b_2, verbose=False)

        classical = LA.solve(pd_2x2, b_2)
        classical_norm = _normalized(classical)
        cosine_sim = np.abs(np.dot(_normalized(solution), classical_norm))
        assert cosine_sim > 0.7, f"Cosine similarity too low: {cosine_sim}"
        assert 0 < success_rate <= 1
        assert np.isfinite(residual)


# ===================================================================
# Integration: process via SwapTestReadout
# ===================================================================

class TestProcessQiskitSwapTest:

    def test_exp_value_in_range(self, aer_backend, pd_2x2, b_2):
        classical = LA.solve(pd_2x2, b_2)
        swap_vec = _normalized(classical)
        result = _run_hhl_2x2_swap_test(aer_backend, pd_2x2, b_2, swap_vec, shots=4096)
        readout = SwapTestReadout(swap_test_vector=swap_vec)
        exp_value, success_rate, residual = readout.process(result, pd_2x2, b_2)

        assert 0 <= exp_value <= 1
        assert success_rate > 0
        assert np.isfinite(residual)
