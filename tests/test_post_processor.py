"""Tests for qlsas.post_processor.Post_Processor."""

import numpy as np
import numpy.linalg as LA
import pytest

from qlsas.post_processor import Post_Processor
from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL
from qlsas.transpiler import Transpiler
from qlsas.executer import Executer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized(v):
    return v / LA.norm(v)


def _run_hhl_2x2_measure_x(aer_backend, A, b, shots=4096):
    """Build, transpile, and execute a 2x2 measure_x HHL circuit; return the raw result."""
    sp = StatePrep(method="default")
    hhl = HHL(state_prep=sp, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
    circ = hhl.build_circuit(A, b)
    transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=1)
    tc = transpiler.optimize()
    ex = Executer()
    return ex.run(tc, aer_backend, shots=shots, verbose=False)


def _run_hhl_2x2_swap_test(aer_backend, A, b, swap_vec, shots=4096):
    """Build, transpile, and execute a 2x2 swap_test HHL circuit; return the raw result."""
    sp = StatePrep(method="default")
    hhl = HHL(state_prep=sp, readout="swap_test", num_qpe_qubits=4, eig_oracle="classical")
    circ = hhl.build_circuit(A, b, swap_test_vector=swap_vec)
    transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=1)
    tc = transpiler.optimize()
    ex = Executer()
    return ex.run(tc, aer_backend, shots=shots, verbose=False)


# ===================================================================
# norm_estimation
# ===================================================================

class TestNormEstimation:

    def test_known_system(self):
        pp = Post_Processor()
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x_true = LA.solve(A, b)
        x_norm = _normalized(x_true)
        alpha = pp.norm_estimation(A, b, x_norm)
        scaled = alpha * x_norm
        assert np.allclose(A @ scaled, b, atol=1e-6)

    def test_zero_denominator(self):
        pp = Post_Processor()
        A = np.array([[0.0, 0.0], [0.0, 0.0]])
        b = np.array([1.0, 0.0])
        x = np.array([0.0, 0.0])
        result = pp.norm_estimation(A, b, x)
        assert result == pytest.approx(1e-10)


# ===================================================================
# tomography_from_counts
# ===================================================================

class TestTomographyFromCounts:

    def test_synthetic_counts_2x2(self):
        """For a 2x2 system, x_size=1. Construct counts where coord 0 gets 75 hits, coord 1 gets 25."""
        pp = Post_Processor()
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = _normalized(np.array([1.0, 1.0]))

        counts = {
            "01": 75,   # coord 0, ancilla=1 (success)
            "11": 25,   # coord 1, ancilla=1 (success)
            "00": 50,   # coord 0, ancilla=0 (fail)
            "10": 50,   # coord 1, ancilla=0 (fail)
        }
        solution, success_rate, residual = pp.tomography_from_counts(counts, A, b)

        assert len(solution) == 2
        assert success_rate == pytest.approx(100 / 200)
        assert np.isfinite(residual)

    def test_no_successful_shots_raises(self):
        pp = Post_Processor()
        A = np.eye(2)
        b = _normalized(np.array([1.0, 0.0]))
        counts = {"00": 50, "10": 50}
        with pytest.raises(ValueError, match="No successful shots"):
            pp.tomography_from_counts(counts, A, b)

    def test_solution_is_unit_normalized(self):
        pp = Post_Processor()
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        b = _normalized(np.array([1.0, 1.0]))
        counts = {"01": 60, "11": 40, "00": 100}
        solution, _, _ = pp.tomography_from_counts(counts, A, b)
        assert np.isclose(LA.norm(solution), 1.0, atol=1e-5)


# ===================================================================
# _finish_tomography — sign correction
# ===================================================================

class TestFinishTomography:

    def test_signs_match_classical_solution(self):
        pp = Post_Processor()
        A = np.array([[2.0, -1.0], [-1.0, 2.0]])
        b = _normalized(np.array([1.0, -1.0]))
        classical = LA.solve(A, b)

        approx_sol = np.abs(_normalized(classical))
        solution, _, _ = pp._finish_tomography(
            approx_sol.copy(), 0.5, 100, 200, A, b
        )
        for i in range(len(solution)):
            assert np.sign(solution[i]) == np.sign(classical[i]) or np.isclose(solution[i], 0, atol=1e-10)


# ===================================================================
# Integration: process_qiskit_tomography
# ===================================================================

class TestProcessQiskitTomography:

    def test_solution_direction_close_to_classical(self, aer_backend, pd_2x2, b_2):
        result = _run_hhl_2x2_measure_x(aer_backend, pd_2x2, b_2, shots=4096)
        pp = Post_Processor()
        solution, success_rate, residual = pp.process_qiskit_tomography(result, pd_2x2, b_2, verbose=False)

        classical = LA.solve(pd_2x2, b_2)
        classical_norm = _normalized(classical)
        cosine_sim = np.abs(np.dot(_normalized(solution), classical_norm))
        assert cosine_sim > 0.7, f"Cosine similarity too low: {cosine_sim}"
        assert 0 < success_rate <= 1
        assert np.isfinite(residual)


# ===================================================================
# Integration: process_qiskit_swap_test
# ===================================================================

class TestProcessQiskitSwapTest:

    def test_exp_value_in_range(self, aer_backend, pd_2x2, b_2):
        classical = LA.solve(pd_2x2, b_2)
        swap_vec = _normalized(classical)
        result = _run_hhl_2x2_swap_test(aer_backend, pd_2x2, b_2, swap_vec, shots=4096)
        pp = Post_Processor()
        exp_value, success_rate, residual = pp.process_qiskit_swap_test(result, pd_2x2, b_2, swap_vec)

        assert 0 <= exp_value <= 1
        assert success_rate > 0
        assert np.isfinite(residual)
