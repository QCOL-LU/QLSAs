"""Tests for qlsas.solver.QuantumLinearSolver."""

import numpy as np
import numpy.linalg as LA
import pytest

from qlsas.backends.base import Backend, CompiledArtifact
from qlsas.measurement_result import MeasurementResult
from qlsas.solver import QuantumLinearSolver, SolveResult
from qlsas.state_prep import StatePrep, DefaultStatePrep
from qlsas.algorithms.hhl import HHL, MCRYEigOracle
from qlsas.executer import Executer
from qlsas.ibm_options import IBMExecutionOptions
from qlsas.readout import MeasureXReadout, SwapTestReadout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized(v):
    return v / LA.norm(v)


def _make_solver(backend, readout=None, num_qpe=4, oracle=None, shots=1024, **kwargs):
    hhl = HHL(num_qpe_qubits=num_qpe, eig_oracle=oracle or MCRYEigOracle())
    rd = readout or MeasureXReadout()
    return QuantumLinearSolver(
        qlsa=hhl,
        readout=rd,
        backend=backend,
        state_prep=DefaultStatePrep(),
        shots=shots,
        **kwargs,
    )


class _StubQLSA:
    def build_circuit(self, A, b, state_prep, **kwargs):
        from qlsas.readout.base import QLSACircuit
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        qr = QuantumRegister(1, "b_to_x_register")
        ar = QuantumRegister(1, "ancilla_flag_register")
        cr = ClassicalRegister(1, "ancilla_flag_result")
        circ = QuantumCircuit(ar, qr, cr)
        return QLSACircuit(circuit=circ, solution_register=qr, ancilla_register=ar, ancilla_creg=cr)


class _StubReadout:
    register_names = ["ancilla_flag_result", "x_result"]

    def apply(self, qlsa_circuit, *, state_prep=None):
        return qlsa_circuit.circuit

    def process(self, result, A, b, verbose=True):
        return np.array([1.0, 0.0]), 1.0, 0.0


class _RecordingBackend(Backend):
    """Test double for the Backend protocol that captures every call.

    Used to verify the solver threads ``ibm_options`` and other run-time
    parameters into the adapter rather than dropping them on the floor.
    """

    def __init__(self):
        self.compile_calls = []
        self.run_calls = []

    @property
    def name(self) -> str:
        return "recording-backend"

    def compile(self, qc, optimization_level=2):
        self.compile_calls.append((qc, optimization_level))
        return CompiledArtifact(payload="fake-transpiled-circuit")

    def run_compiled(self, artifact, shots=1024, *, verbose=True, **kwargs):
        self.run_calls.append(
            {
                "artifact": artifact,
                "shots": shots,
                "ibm_options": kwargs.get("ibm_options"),
                "session": kwargs.get("session"),
            }
        )
        return MeasurementResult("fake-result")


# ===================================================================
# Constructor defaults
# ===================================================================

class TestConstructorDefaults:

    def test_shots_per_batch_defaults_to_shots(self, aer_backend):
        solver = _make_solver(aer_backend, shots=512)
        assert solver.shots_per_batch == 512

    def test_executer_default(self, aer_backend):
        solver = _make_solver(aer_backend)
        assert isinstance(solver.executer, Executer)

    def test_custom_executer_used(self, aer_backend):
        custom_ex = Executer()
        solver = _make_solver(aer_backend, executer=custom_ex)
        assert solver.executer is custom_ex

    def test_default_executer_receives_ibm_options(self, aer_backend):
        ibm_options = IBMExecutionOptions(enable_error_mitigation=True)
        solver = _make_solver(aer_backend, ibm_options=ibm_options)
        assert solver.executer.ibm_options is ibm_options

    def test_default_state_prep_created(self, aer_backend):
        hhl = HHL(num_qpe_qubits=4)
        rd = MeasureXReadout()
        solver = QuantumLinearSolver(qlsa=hhl, readout=rd, backend=aer_backend)
        assert isinstance(solver.state_prep, StatePrep)


class TestIBMOptionsThreading:

    def test_solver_passes_ibm_options_to_backend(self):
        ibm_options = IBMExecutionOptions(
            enable_error_mitigation=True,
            enable_dynamical_decoupling=True,
        )
        backend = _RecordingBackend()
        solver = QuantumLinearSolver(
            qlsa=_StubQLSA(),
            readout=_StubReadout(),
            backend=backend,
            state_prep=DefaultStatePrep(),
            shots=128,
            ibm_options=ibm_options,
        )

        result = solver.solve(np.eye(2), np.array([1.0, 0.0]), verbose=False)

        assert isinstance(result, SolveResult)
        assert len(backend.run_calls) == 1
        assert backend.run_calls[0]["ibm_options"] is ibm_options


# ===================================================================
# End-to-end solve — measure_x
# ===================================================================

class TestSolveMeasureX:

    def test_2x2_pd_solution_direction(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend, shots=4096)
        result = solver.solve(pd_2x2, b_2, verbose=False)

        assert isinstance(result, SolveResult)
        classical = LA.solve(pd_2x2, b_2)
        cosine_sim = np.abs(np.dot(_normalized(result.solution), _normalized(classical)))
        assert cosine_sim > 0.7, f"Cosine similarity too low: {cosine_sim}"

    def test_2x2_indefinite(self, aer_backend, indefinite_2x2, b_2):
        solver = _make_solver(aer_backend, shots=4096)
        result = solver.solve(indefinite_2x2, b_2, verbose=False)
        assert result.solution.shape == (2,)
        assert np.all(np.isfinite(result.solution))

    def test_2x2_diagonal(self, aer_backend, diagonal_2x2, b_2):
        solver = _make_solver(aer_backend, shots=4096)
        result = solver.solve(diagonal_2x2, b_2, verbose=False)

        classical = LA.solve(diagonal_2x2, b_2)
        cosine_sim = np.abs(np.dot(_normalized(result.solution), _normalized(classical)))
        assert cosine_sim > 0.7

    def test_2x2_identity(self, aer_backend, identity_2x2, b_2):
        solver = _make_solver(aer_backend, shots=4096)
        result = solver.solve(identity_2x2, b_2, verbose=False)

        cosine_sim = np.abs(np.dot(_normalized(result.solution), _normalized(b_2)))
        assert cosine_sim > 0.7

    @pytest.mark.slow
    def test_4x4_pd(self, aer_backend, pd_4x4, b_4):
        solver = _make_solver(aer_backend, shots=4096)
        result = solver.solve(pd_4x4, b_4, verbose=False)
        assert result.solution.shape == (4,)
        assert np.all(np.isfinite(result.solution))

    @pytest.mark.slow
    def test_8x8_pd(self, aer_backend, pd_8x8, b_8):
        solver = _make_solver(aer_backend, shots=4096)
        result = solver.solve(pd_8x8, b_8, verbose=False)
        assert result.solution.shape == (8,)
        assert np.all(np.isfinite(result.solution))


# ===================================================================
# End-to-end solve — swap_test
# ===================================================================

class TestSolveSwapTest:

    def test_swap_test_runs_through_solver(self, aer_backend, pd_2x2, b_2):
        classical = LA.solve(pd_2x2, b_2)
        swap_vec = _normalized(classical)
        readout = SwapTestReadout(swap_test_vector=swap_vec)
        solver = _make_solver(aer_backend, readout=readout, shots=4096)
        # Should run without error — swap_test_vector is on the readout
        result = solver.solve(pd_2x2, b_2, verbose=False)
        assert isinstance(result, SolveResult)
        assert np.isfinite(result.solution)


# ===================================================================
# target_successful_shots
# ===================================================================

class TestTargetSuccessfulShots:

    def test_accumulates_to_target(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(
            aer_backend,
            shots=1024,
            target_successful_shots=10,
            shots_per_batch=50,
        )
        result = solver.solve(pd_2x2, b_2, verbose=False)
        assert isinstance(result, SolveResult)
        assert result.solution.shape == (2,)
        assert np.all(np.isfinite(result.solution))
        assert result.success_rate is not None

    def test_max_total_shots_cap(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(
            aer_backend,
            shots=1024,
            target_successful_shots=100_000,
            shots_per_batch=50,
            max_total_shots=200,
        )
        result = solver.solve(pd_2x2, b_2, verbose=False)
        assert result.solution.shape == (2,)
