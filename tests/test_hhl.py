"""Tests for qlsas.algorithms.hhl.hhl.HHL."""

import math
import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized(v):
    return v / np.linalg.norm(v)


def _register_names(circuit: QuantumCircuit) -> set[str]:
    qnames = {r.name for r in circuit.qregs}
    cnames = {r.name for r in circuit.cregs}
    return qnames | cnames


# ===================================================================
# Circuit structure — measure_x readout
# ===================================================================

class TestMeasureXCircuit:

    @pytest.mark.parametrize("fixture_A, fixture_b, oracle", [
        ("pd_2x2", "b_2", "classical"),
        ("pd_2x2", "b_2", "quantum"),
        ("pd_4x4", "b_4", "classical"),
        ("indefinite_2x2", "b_2", "classical"),
    ])
    def test_register_names(self, fixture_A, fixture_b, oracle, state_prep, request):
        A = request.getfixturevalue(fixture_A)
        b = request.getfixturevalue(fixture_b)
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle=oracle)
        circ = hhl.build_circuit(A, b)

        names = _register_names(circ)
        assert "ancilla_flag_register" in names
        assert "qpe_register" in names
        assert "b_to_x_register" in names
        assert "ancilla_flag_result" in names
        assert "x_result" in names

    @pytest.mark.parametrize("num_qpe, size, fixture_A, fixture_b", [
        (3, 2, "pd_2x2", "b_2"),
        (4, 2, "pd_2x2", "b_2"),
        (4, 4, "pd_4x4", "b_4"),
    ])
    def test_qubit_count(self, num_qpe, size, fixture_A, fixture_b, state_prep, request):
        A = request.getfixturevalue(fixture_A)
        b = request.getfixturevalue(fixture_b)
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=num_qpe, eig_oracle="classical")
        circ = hhl.build_circuit(A, b)
        expected = 1 + num_qpe + int(math.log2(size))
        assert circ.num_qubits == expected

    @pytest.mark.slow
    def test_8x8_circuit_builds(self, pd_8x8, b_8, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        circ = hhl.build_circuit(pd_8x8, b_8)
        expected = 1 + 4 + 3  # ancilla + qpe + log2(8)
        assert circ.num_qubits == expected

    def test_circuit_name_measure_x(self, pd_2x2, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        circ = hhl.build_circuit(pd_2x2, b_2)
        assert "HHL" in circ.name
        assert "2 by 2" in circ.name


# ===================================================================
# Circuit structure — swap_test readout
# ===================================================================

class TestSwapTestCircuit:

    def test_register_names(self, pd_2x2, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="swap_test", num_qpe_qubits=4, eig_oracle="classical")
        swap_vec = _normalized(np.array([1.0, 0.0]))
        circ = hhl.build_circuit(pd_2x2, b_2, swap_test_vector=swap_vec)

        names = _register_names(circ)
        assert "swap_test_ancilla_register" in names
        assert "v_register" in names
        assert "swap_test_result" in names

    def test_qubit_count_swap_test(self, pd_2x2, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="swap_test", num_qpe_qubits=4, eig_oracle="classical")
        swap_vec = _normalized(np.array([1.0, 0.0]))
        circ = hhl.build_circuit(pd_2x2, b_2, swap_test_vector=swap_vec)
        data_qubits = int(math.log2(2))
        expected = 1 + 4 + data_qubits + 1 + data_qubits
        assert circ.num_qubits == expected

    def test_circuit_name_swap_test(self, pd_2x2, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="swap_test", num_qpe_qubits=4, eig_oracle="classical")
        swap_vec = _normalized(np.array([1.0, 0.0]))
        circ = hhl.build_circuit(pd_2x2, b_2, swap_test_vector=swap_vec)
        assert "Swap Test" in circ.name


# ===================================================================
# Input validation
# ===================================================================

class TestBuildCircuitValidation:

    def test_non_square_A(self, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        A = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="square"):
            hhl.build_circuit(A, b_2)

    def test_non_vector_b(self, pd_2x2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="vector"):
            hhl.build_circuit(pd_2x2, b)

    def test_dimension_mismatch(self, pd_4x4, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        with pytest.raises(ValueError, match="mismatch"):
            hhl.build_circuit(pd_4x4, b_2)

    def test_non_hermitian(self, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        A = np.array([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="Hermitian"):
            hhl.build_circuit(A, b_2)

    def test_non_unit_norm_b(self, pd_2x2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        b = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="unit norm"):
            hhl.build_circuit(pd_2x2, b)

    def test_invalid_readout(self, state_prep):
        hhl = HHL(state_prep=state_prep, readout="invalid", num_qpe_qubits=4, eig_oracle="classical")
        A = np.eye(2)
        b = _normalized(np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="readout"):
            hhl.build_circuit(A, b)

    def test_invalid_eig_oracle(self, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="bad")
        A = np.eye(2)
        b = _normalized(np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="eig_oracle"):
            hhl.build_circuit(A, b)

    def test_swap_test_without_vector(self, pd_2x2, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="swap_test", num_qpe_qubits=4, eig_oracle="classical")
        with pytest.raises(ValueError, match="swap_test_vector"):
            hhl.build_circuit(pd_2x2, b_2)

    def test_measure_x_with_swap_vector_warns(self, pd_2x2, b_2, state_prep):
        hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")
        swap_vec = _normalized(np.array([1.0, 0.0]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hhl.build_circuit(pd_2x2, b_2, swap_test_vector=swap_vec)
            assert any("swap_test_vector" in str(warning.message) for warning in w)


# ===================================================================
# Parameterized across QPE qubits and oracles
# ===================================================================

@pytest.mark.parametrize("num_qpe", [3, 4])
@pytest.mark.parametrize("oracle", ["classical", "quantum"])
def test_builds_for_various_configs(pd_2x2, b_2, state_prep, num_qpe, oracle):
    hhl = HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=num_qpe, eig_oracle=oracle)
    circ = hhl.build_circuit(pd_2x2, b_2)
    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == 1 + num_qpe + 1  # ancilla + qpe + 1 data qubit for 2x2
