"""Tests for qlsas.algorithms.hhl.hhl_helpers."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, ExactReciprocalGate

from qlsas.algorithms.hhl.hhl_helpers import (
    classical_eig_inversion_oracle,
    quantum_eig_inversion_oracle,
    dynamic_t0,
    C_factor,
)


# ===================================================================
# dynamic_t0
# ===================================================================

class TestDynamicT0:

    def test_positive_float(self, pd_2x2):
        t0 = dynamic_t0(pd_2x2)
        assert isinstance(t0, float)
        assert t0 > 0

    @pytest.mark.parametrize("fixture_name", ["pd_2x2", "indefinite_2x2", "diagonal_2x2", "pd_4x4", "pd_8x8"])
    def test_no_aliasing(self, fixture_name, request):
        A = request.getfixturevalue(fixture_name)
        t0 = dynamic_t0(A)
        max_abs_eig = np.max(np.abs(np.linalg.eigvalsh(A)))
        assert t0 * max_abs_eig < np.pi

    def test_buffer_respected(self, pd_2x2):
        t0_default = dynamic_t0(pd_2x2)
        t0_tight = dynamic_t0(pd_2x2, buffer=0.0)
        assert t0_tight > t0_default

    def test_identity_matrix(self, identity_2x2):
        t0 = dynamic_t0(identity_2x2)
        assert t0 > 0
        assert t0 * 1.0 < np.pi


# ===================================================================
# C_factor
# ===================================================================

class TestCFactor:

    def test_less_than_min_eig(self, pd_2x2):
        C = C_factor(pd_2x2)
        min_abs_eig = np.min(np.abs(np.linalg.eigvalsh(pd_2x2)))
        assert C < min_abs_eig

    def test_positive(self, pd_2x2):
        assert C_factor(pd_2x2) > 0

    def test_scale_parameter(self, pd_2x2):
        C_high = C_factor(pd_2x2, scale=0.95)
        C_low = C_factor(pd_2x2, scale=0.5)
        assert C_high > C_low

    def test_near_singular_filters_zero_eigs(self, near_singular_2x2):
        C = C_factor(near_singular_2x2)
        assert C > 0
        assert np.isfinite(C)

    def test_indefinite_matrix(self, indefinite_2x2):
        C = C_factor(indefinite_2x2)
        abs_eigs = np.abs(np.linalg.eigvalsh(indefinite_2x2))
        assert C < np.min(abs_eigs[abs_eigs > 1e-12])


# ===================================================================
# classical_eig_inversion_oracle
# ===================================================================

class TestClassicalOracle:

    @pytest.mark.parametrize("num_qpe_qubits", [2, 3, 4])
    def test_appends_controlled_ry_gates(self, pd_2x2, num_qpe_qubits):
        qpe_reg = QuantumRegister(num_qpe_qubits, name="qpe")
        anc = QuantumRegister(1, name="anc")
        circ = QuantumCircuit(qpe_reg, anc)
        depth_before = circ.depth()

        t0 = dynamic_t0(pd_2x2)
        C = C_factor(pd_2x2)
        classical_eig_inversion_oracle(circ, qpe_reg, anc[0], A=pd_2x2, t0=t0, C=C)

        assert circ.depth() > depth_before
        ry_count = sum(1 for inst in circ.data if isinstance(inst.operation, RYGate) or "ry" in inst.operation.name.lower())
        assert ry_count >= 2**num_qpe_qubits

    @pytest.mark.parametrize("fixture_name", ["pd_2x2", "indefinite_2x2"])
    def test_no_error_pd_and_indefinite(self, fixture_name, request):
        A = request.getfixturevalue(fixture_name)
        qpe_reg = QuantumRegister(3, name="qpe")
        anc = QuantumRegister(1, name="anc")
        circ = QuantumCircuit(qpe_reg, anc)
        t0 = dynamic_t0(A)
        C = C_factor(A)
        classical_eig_inversion_oracle(circ, qpe_reg, anc[0], A=A, t0=t0, C=C)


# ===================================================================
# quantum_eig_inversion_oracle
# ===================================================================

class TestQuantumOracle:

    def test_appends_exact_reciprocal_gate(self, pd_2x2):
        qpe_reg = QuantumRegister(3, name="qpe")
        anc = QuantumRegister(1, name="anc")
        circ = QuantumCircuit(qpe_reg, anc)
        depth_before = circ.depth()

        t0 = dynamic_t0(pd_2x2)
        C = C_factor(pd_2x2)
        quantum_eig_inversion_oracle(circ, qpe_reg, anc[0], A=pd_2x2, t0=t0, C=C)

        assert circ.depth() > depth_before

    def test_neg_vals_flag_positive_definite(self, pd_2x2):
        """For a PD matrix, neg_vals should be False inside the oracle."""
        eigs = np.linalg.eigvalsh(pd_2x2)
        has_negative = bool(np.any(eigs < -1e-12))
        assert not has_negative

    def test_neg_vals_flag_indefinite(self, indefinite_2x2):
        """For an indefinite matrix, neg_vals should be True."""
        eigs = np.linalg.eigvalsh(indefinite_2x2)
        has_negative = bool(np.any(eigs < -1e-12))
        assert has_negative

    @pytest.mark.parametrize("fixture_name", ["pd_2x2", "indefinite_2x2"])
    def test_no_error_pd_and_indefinite(self, fixture_name, request):
        A = request.getfixturevalue(fixture_name)
        qpe_reg = QuantumRegister(3, name="qpe")
        anc = QuantumRegister(1, name="anc")
        circ = QuantumCircuit(qpe_reg, anc)
        t0 = dynamic_t0(A)
        C = C_factor(A)
        quantum_eig_inversion_oracle(circ, qpe_reg, anc[0], A=A, t0=t0, C=C)
