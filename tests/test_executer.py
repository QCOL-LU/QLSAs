"""Tests for qlsas.executer.Executer."""

import numpy as np
import pytest
from pytket.circuit import Circuit as TketCircuit
from qnexus import QuantinuumConfig

from qlsas.executer import Executer
from qlsas.transpiler import Transpiler
from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transpiled_2x2_circuit(aer_backend):
    """Build and transpile a 2x2 HHL circuit for the AerSimulator."""
    sp = StatePrep(method="default")
    hhl = HHL(state_prep=sp, readout="measure_x", num_qpe_qubits=3, eig_oracle="classical")
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 0.0])
    circ = hhl.build_circuit(A, b)
    transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=1)
    return transpiler.optimize()


# ===================================================================
# Session lifecycle
# ===================================================================

class TestSessionLifecycle:

    def test_open_session_aer_is_noop(self, aer_backend):
        ex = Executer()
        ex.open_session(aer_backend, verbose=False)
        assert not ex.session_active

    def test_close_session_when_none_is_noop(self):
        ex = Executer()
        ex.close_session(verbose=False)
        assert not ex.session_active

    def test_double_open_is_idempotent(self, aer_backend):
        ex = Executer()
        ex.open_session(aer_backend, verbose=False)
        ex.open_session(aer_backend, verbose=False)
        assert not ex.session_active

    def test_context_manager_aer(self, aer_backend):
        ex = Executer()
        with ex.session(aer_backend, verbose=False):
            assert not ex.session_active
        assert not ex.session_active


# ===================================================================
# Execution
# ===================================================================

class TestExecution:

    def test_run_aer_returns_result(self, aer_backend):
        tc = _transpiled_2x2_circuit(aer_backend)
        ex = Executer()
        result = ex.run(tc, aer_backend, shots=100, verbose=False)
        assert result is not None

    def test_result_has_join_data(self, aer_backend):
        tc = _transpiled_2x2_circuit(aer_backend)
        ex = Executer()
        result = ex.run(tc, aer_backend, shots=100, verbose=False)
        joined = result.join_data(names=["ancilla_flag_result", "x_result"])
        counts = joined.get_counts()
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 100

    def test_run_qnexus_not_implemented(self):
        ex = Executer()
        backend = QuantinuumConfig(device_name="H1-1LE")
        tket_circ = TketCircuit(2)
        tket_circ.H(0)
        with pytest.raises(NotImplementedError):
            ex.run(tket_circ, backend, shots=10, verbose=False)

    def test_invalid_backend_type(self):
        ex = Executer()
        with pytest.raises(ValueError, match="backend type"):
            ex.run(None, {"bad": True}, shots=10, verbose=False)
