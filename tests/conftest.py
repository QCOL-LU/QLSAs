"""Shared fixtures for the qlsas test suite."""

import numpy as np
import pytest
from qiskit_aer import AerSimulator

from qlsas.data_loader import StatePrep
from qlsas.algorithms.hhl.hhl import HHL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spd(n: int, cond: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive-definite matrix with a target condition number."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(1.0, cond, n)
    return (Q * eigs) @ Q.T


def _normalized(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def aer_backend():
    return AerSimulator()


# ---------------------------------------------------------------------------
# StatePrep
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def state_prep():
    return StatePrep(method="default")


# ---------------------------------------------------------------------------
# 2x2 matrices
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def identity_2x2():
    return np.eye(2)


@pytest.fixture(scope="session")
def diagonal_2x2():
    return np.diag([2.0, 0.5])


@pytest.fixture(scope="session")
def pd_2x2():
    return np.array([[2.0, 1.0], [1.0, 3.0]])


@pytest.fixture(scope="session")
def indefinite_2x2():
    return np.array([[1.0, 0.0], [0.0, -2.0]])


@pytest.fixture(scope="session")
def near_singular_2x2():
    return np.array([[1.0, 0.0], [0.0, 1e-6]])


# ---------------------------------------------------------------------------
# 4x4 matrices
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pd_4x4():
    rng = np.random.default_rng(42)
    return _random_spd(4, cond=5.0, rng=rng)


@pytest.fixture(scope="session")
def ill_conditioned_4x4():
    rng = np.random.default_rng(43)
    return _random_spd(4, cond=1000.0, rng=rng)


@pytest.fixture(scope="session")
def indefinite_4x4():
    rng = np.random.default_rng(44)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    eigs = np.array([-2.0, -0.5, 1.0, 3.0])
    return (Q * eigs) @ Q.T


# ---------------------------------------------------------------------------
# 8x8 matrices
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pd_8x8():
    rng = np.random.default_rng(45)
    return _random_spd(8, cond=10.0, rng=rng)


# ---------------------------------------------------------------------------
# Normalized b vectors
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def b_2():
    rng = np.random.default_rng(100)
    v = rng.standard_normal(2)
    return _normalized(v)


@pytest.fixture(scope="session")
def b_4():
    rng = np.random.default_rng(101)
    v = rng.standard_normal(4)
    return _normalized(v)


@pytest.fixture(scope="session")
def b_8():
    rng = np.random.default_rng(102)
    v = rng.standard_normal(8)
    return _normalized(v)


# ---------------------------------------------------------------------------
# HHL algorithm configurations
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hhl_measure_x_classical(state_prep):
    return HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="classical")


@pytest.fixture(scope="session")
def hhl_measure_x_quantum(state_prep):
    return HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=4, eig_oracle="quantum")


@pytest.fixture(scope="session")
def hhl_swap_test_classical(state_prep):
    return HHL(state_prep=state_prep, readout="swap_test", num_qpe_qubits=4, eig_oracle="classical")


@pytest.fixture(scope="session")
def hhl_3qpe_measure_x(state_prep):
    return HHL(state_prep=state_prep, readout="measure_x", num_qpe_qubits=3, eig_oracle="classical")
