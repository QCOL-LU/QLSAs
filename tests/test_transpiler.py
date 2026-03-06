"""Tests for qlsas.transpiler.Transpiler."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qnexus import QuantinuumConfig
from pytket.circuit import Circuit as TketCircuit

from qlsas.transpiler import Transpiler
from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_hhl_circuit() -> QuantumCircuit:
    """Build a small 2x2 HHL circuit for transpilation tests."""
    sp = StatePrep(method="default")
    hhl = HHL(state_prep=sp, readout="measure_x", num_qpe_qubits=3, eig_oracle="classical")
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 0.0])
    return hhl.build_circuit(A, b)


# ===================================================================
# Qiskit transpilation path
# ===================================================================

class TestQiskitTranspilation:

    @pytest.mark.parametrize("opt_level", [0, 1, 2, 3])
    def test_all_optimization_levels(self, aer_backend, opt_level):
        circ = _small_hhl_circuit()
        transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=opt_level)
        result = transpiler.optimize()
        assert isinstance(result, QuantumCircuit)

    def test_output_has_qubits(self, aer_backend):
        circ = _small_hhl_circuit()
        transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=1)
        result = transpiler.optimize()
        assert result.num_qubits > 0


# ===================================================================
# pytket Circuit input
# ===================================================================

class TestPytketInput:

    def test_tket_circuit_converted_and_transpiled(self, aer_backend):
        tket_circ = TketCircuit(2)
        tket_circ.H(0)
        tket_circ.CX(0, 1)
        transpiler = Transpiler(circuit=tket_circ, backend=aer_backend, optimization_level=1)
        result = transpiler.optimize()
        assert isinstance(result, QuantumCircuit)


# ===================================================================
# Validation errors
# ===================================================================

class TestTranspilerValidation:

    def test_invalid_optimization_level(self, aer_backend):
        circ = _small_hhl_circuit()
        transpiler = Transpiler(circuit=circ, backend=aer_backend, optimization_level=5)
        with pytest.raises(ValueError, match="optimization level"):
            transpiler.optimize()

    def test_invalid_circuit_type(self, aer_backend):
        transpiler = Transpiler(circuit="not_a_circuit", backend=aer_backend, optimization_level=1)
        with pytest.raises(ValueError, match="circuit type"):
            transpiler.optimize()

    def test_quantinuum_backend_not_implemented(self):
        circ = _small_hhl_circuit()
        backend = QuantinuumConfig(device_name="H1-1LE")
        transpiler = Transpiler(circuit=circ, backend=backend, optimization_level=1)
        with pytest.raises(NotImplementedError):
            transpiler.optimize()

    def test_invalid_backend_type(self):
        circ = _small_hhl_circuit()
        transpiler = Transpiler(circuit=circ, backend={"fake": True}, optimization_level=1)
        with pytest.raises(ValueError, match="backend type"):
            transpiler.optimize()
