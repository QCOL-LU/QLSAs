"""Tests for qlsas.refiner.Refiner."""

import numpy as np
import numpy.linalg as LA
import pytest

from qlsas.refiner import Refiner
from qlsas.solver import QuantumLinearSolver
from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized(v):
    return v / LA.norm(v)


def _make_solver(aer_backend, shots=2048):
    sp = StatePrep(method="default")
    hhl = HHL(state_prep=sp, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
    return QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=shots)


# ===================================================================
# norm_estimation (Refiner's own copy)
# ===================================================================

class TestRefinerNormEstimation:

    def test_known_system(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend)
        refiner = Refiner(A=pd_2x2, b=b_2, solver=solver)
        x_true = LA.solve(pd_2x2, b_2)
        x_norm = _normalized(x_true)
        alpha = refiner.norm_estimation(pd_2x2, b_2, x_norm)
        scaled = alpha * x_norm
        assert np.allclose(pd_2x2 @ scaled, b_2, atol=1e-6)

    def test_zero_denominator(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend)
        refiner = Refiner(A=pd_2x2, b=b_2, solver=solver)
        A_zero = np.zeros((2, 2))
        x_zero = np.zeros(2)
        result = refiner.norm_estimation(A_zero, b_2, x_zero)
        assert result == pytest.approx(1e-10)


# ===================================================================
# refine integration
# ===================================================================

class TestRefineIntegration:

    def test_convergence_2x2(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend, shots=2048)
        refiner = Refiner(A=pd_2x2, b=b_2, solver=solver)
        result = refiner.refine(
            precision=0.5,
            max_iter=3,
            plot=False,
            verbose=False,
        )
        assert result["residuals"][-1] < LA.norm(b_2)

    def test_returned_dict_keys(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend, shots=1024)
        refiner = Refiner(A=pd_2x2, b=b_2, solver=solver)
        result = refiner.refine(
            precision=0.01,
            max_iter=1,
            plot=False,
            verbose=False,
        )
        expected_keys = {"refined_x", "residuals", "errors", "total_iterations", "initial_solution", "transpiled_circuits"}
        assert expected_keys == set(result.keys())

    def test_plot_false_no_error(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend, shots=1024)
        refiner = Refiner(A=pd_2x2, b=b_2, solver=solver)
        refiner.refine(precision=0.01, max_iter=1, plot=False, verbose=False)

    def test_max_iter_zero(self, aer_backend, pd_2x2, b_2):
        """max_iter=0 means the while condition allows iteration 0, so exactly 1 iteration runs."""
        solver = _make_solver(aer_backend, shots=1024)
        refiner = Refiner(A=pd_2x2, b=b_2, solver=solver)
        result = refiner.refine(
            precision=0.001,
            max_iter=0,
            plot=False,
            verbose=False,
        )
        assert result["total_iterations"] <= 1

    @pytest.mark.slow
    def test_convergence_4x4(self, aer_backend, pd_4x4, b_4):
        solver = _make_solver(aer_backend, shots=2048)
        refiner = Refiner(A=pd_4x4, b=b_4, solver=solver)
        result = refiner.refine(
            precision=0.5,
            max_iter=3,
            plot=False,
            verbose=False,
        )
        assert len(result["residuals"]) >= 1
