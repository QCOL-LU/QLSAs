"""Tests for qlsas.solver.QuantumLinearSolver."""

import numpy as np
import numpy.linalg as LA
import pytest

from qlsas.solver import QuantumLinearSolver
from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL
from qlsas.executer import Executer
from qlsas.ibm_options import IBMExecutionOptions
from qlsas.post_processor import Post_Processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized(v):
    return v / LA.norm(v)


def _make_hhl(readout="measure_x", num_qpe=4, oracle="classical"):
    sp = StatePrep(method="default")
    return HHL(state_prep=sp, readout=readout, num_qpe_qubits=num_qpe, eig_oracle=oracle)


class _StubQLSA:
    readout = "measure_x"

    def build_circuit(self, A, b, t0=None, C=None):
        return "fake-circuit"


class _RecordingExecuter:
    def __init__(self):
        self.session_active = False
        self.calls = []

    def run(self, transpiled_circuit, backend, shots, ibm_options=None, verbose=True):
        self.calls.append(
            {
                "transpiled_circuit": transpiled_circuit,
                "backend": backend,
                "shots": shots,
                "ibm_options": ibm_options,
            }
        )
        return "fake-result"


class _RecordingPostProcessor:
    def process_tomography(self, result, A, b, verbose=True):
        return np.array([1.0, 0.0]), 1.0, 0.0


class _FakeTranspiler:
    def __init__(self, circuit, backend, optimization_level):
        self.circuit = circuit
        self.backend = backend
        self.optimization_level = optimization_level

    def optimize(self):
        return "fake-transpiled-circuit"


# ===================================================================
# Constructor defaults
# ===================================================================

class TestConstructorDefaults:

    def test_shots_per_batch_defaults_to_shots(self, aer_backend):
        hhl = _make_hhl()
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=512)
        assert solver.shots_per_batch == 512

    def test_executer_default(self, aer_backend):
        hhl = _make_hhl()
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend)
        assert isinstance(solver.executer, Executer)

    def test_post_processor_default(self, aer_backend):
        hhl = _make_hhl()
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend)
        assert isinstance(solver.post_processor, Post_Processor)

    def test_custom_executer_used(self, aer_backend):
        hhl = _make_hhl()
        custom_ex = Executer()
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, executer=custom_ex)
        assert solver.executer is custom_ex

    def test_default_executer_receives_ibm_options(self, aer_backend):
        hhl = _make_hhl()
        ibm_options = IBMExecutionOptions(enable_error_mitigation=True)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, ibm_options=ibm_options)
        assert solver.executer.ibm_options is ibm_options


class TestIBMOptionsThreading:

    def test_solver_passes_ibm_options_to_executer(self, monkeypatch):
        monkeypatch.setattr("qlsas.solver.Transpiler", _FakeTranspiler)

        ibm_options = IBMExecutionOptions(
            enable_error_mitigation=True,
            enable_dynamical_decoupling=True,
        )
        executer = _RecordingExecuter()
        post_processor = _RecordingPostProcessor()
        solver = QuantumLinearSolver(
            qlsa=_StubQLSA(),
            backend=object(),
            shots=128,
            ibm_options=ibm_options,
            executer=executer,
            post_processor=post_processor,
        )

        solver.solve(np.eye(2), np.array([1.0, 0.0]), verbose=False)

        assert len(executer.calls) == 1
        assert executer.calls[0]["ibm_options"] is ibm_options


# ===================================================================
# End-to-end solve — measure_x
# ===================================================================

class TestSolveMeasureX:

    def test_2x2_pd_solution_direction(self, aer_backend, pd_2x2, b_2):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=4096)
        solution = solver.solve(pd_2x2, b_2, verbose=False)

        classical = LA.solve(pd_2x2, b_2)
        cosine_sim = np.abs(np.dot(_normalized(solution), _normalized(classical)))
        assert cosine_sim > 0.7, f"Cosine similarity too low: {cosine_sim}"

    def test_2x2_indefinite(self, aer_backend, indefinite_2x2, b_2):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=4096)
        solution = solver.solve(indefinite_2x2, b_2, verbose=False)
        assert solution.shape == (2,)
        assert np.all(np.isfinite(solution))

    def test_2x2_diagonal(self, aer_backend, diagonal_2x2, b_2):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=4096)
        solution = solver.solve(diagonal_2x2, b_2, verbose=False)

        classical = LA.solve(diagonal_2x2, b_2)
        cosine_sim = np.abs(np.dot(_normalized(solution), _normalized(classical)))
        assert cosine_sim > 0.7

    def test_2x2_identity(self, aer_backend, identity_2x2, b_2):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=4096)
        solution = solver.solve(identity_2x2, b_2, verbose=False)

        cosine_sim = np.abs(np.dot(_normalized(solution), _normalized(b_2)))
        assert cosine_sim > 0.7

    @pytest.mark.slow
    def test_4x4_pd(self, aer_backend, pd_4x4, b_4):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=4096)
        solution = solver.solve(pd_4x4, b_4, verbose=False)
        assert solution.shape == (4,)
        assert np.all(np.isfinite(solution))

    @pytest.mark.slow
    def test_8x8_pd(self, aer_backend, pd_8x8, b_8):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=4096)
        solution = solver.solve(pd_8x8, b_8, verbose=False)
        assert solution.shape == (8,)
        assert np.all(np.isfinite(solution))


# ===================================================================
# End-to-end solve — swap_test
# ===================================================================

class TestSolveSwapTest:

    def test_swap_test_through_solver_raises_without_vector(self, aer_backend, pd_2x2, b_2):
        """solver.solve() does not forward swap_test_vector to build_circuit,
        so the swap_test readout path currently raises ValueError."""
        hhl = _make_hhl(readout="swap_test", num_qpe=4)
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=4096)
        with pytest.raises(ValueError, match="swap_test_vector"):
            solver.solve(pd_2x2, b_2, verbose=False)


# ===================================================================
# Invalid readout
# ===================================================================

class TestInvalidReadout:

    def test_invalid_readout_raises(self, aer_backend, pd_2x2, b_2):
        sp = StatePrep(method="default")
        hhl = HHL(state_prep=sp, readout="bad_readout", num_qpe_qubits=4, eig_oracle="classical")
        solver = QuantumLinearSolver(qlsa=hhl, backend=aer_backend, shots=100)
        with pytest.raises(ValueError, match="readout"):
            solver.solve(pd_2x2, b_2, verbose=False)


# ===================================================================
# target_successful_shots
# ===================================================================

class TestTargetSuccessfulShots:

    def test_accumulates_to_target(self, aer_backend, pd_2x2, b_2):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(
            qlsa=hhl,
            backend=aer_backend,
            shots=1024,
            target_successful_shots=10,
            shots_per_batch=50,
        )
        solution = solver.solve(pd_2x2, b_2, verbose=False)
        assert solution.shape == (2,)
        assert np.all(np.isfinite(solution))

    def test_max_total_shots_cap(self, aer_backend, pd_2x2, b_2):
        hhl = _make_hhl(readout="measure_x", num_qpe=4)
        solver = QuantumLinearSolver(
            qlsa=hhl,
            backend=aer_backend,
            shots=1024,
            target_successful_shots=100_000,
            shots_per_batch=50,
            max_total_shots=200,
        )
        solution = solver.solve(pd_2x2, b_2, verbose=False)
        assert solution.shape == (2,)
