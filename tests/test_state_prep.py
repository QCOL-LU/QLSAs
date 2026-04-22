"""Tests for qlsas.state_prep.StatePrep and DefaultStatePrep."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlsas.state_prep import StatePrep, DefaultStatePrep


# ---------------------------------------------------------------------------
# Valid state loading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", [2, 4, 8])
def test_load_state_returns_circuit_with_correct_qubit_count(state_prep, size):
    state = np.ones(size) / np.sqrt(size)
    circuit = state_prep.load_state(state)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == int(math.log2(size))


@pytest.mark.parametrize("size", [2, 4, 8])
def test_statevector_correctness(state_prep, size):
    """Simulate the prep circuit and check the output matches the input up to global phase."""
    rng = np.random.default_rng(200 + size)
    state = rng.standard_normal(size)
    state = state / np.linalg.norm(state)

    circuit = state_prep.load_state(state)
    sv = np.asarray(Statevector(circuit))

    overlap = np.abs(np.vdot(state, sv))
    assert overlap == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_non_power_of_two_raises(state_prep):
    with pytest.raises(ValueError, match="power of two"):
        state_prep.load_state(np.array([1.0, 0.0, 0.0]))


def test_non_unit_norm_raises(state_prep):
    with pytest.raises(ValueError, match="unit norm"):
        state_prep.load_state(np.array([2.0, 0.0]))


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------

def test_state_prep_is_abstract():
    """StatePrep is an ABC and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StatePrep()  # type: ignore[abstract]


def test_default_state_prep_is_concrete():
    """DefaultStatePrep can be instantiated and used."""
    sp = DefaultStatePrep()
    state = np.array([1.0, 0.0])
    circuit = sp.load_state(state)
    assert isinstance(circuit, QuantumCircuit)
