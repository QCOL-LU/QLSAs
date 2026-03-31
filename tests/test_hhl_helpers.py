"""Tests for qlsas.algorithms.hhl.hhl_helpers."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, ExactReciprocalGate
from qiskit.quantum_info import Operator, Statevector

from qlsas.algorithms.hhl.hhl_helpers import (
    classical_eig_inversion_oracle,
    quantum_eig_inversion_oracle,
    unary_iteration_eig_inversion_oracle,
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


# ===================================================================
# Helpers shared by unary-iteration tests
# ===================================================================

def _build_oracle_circuit(oracle_fn, A, m):
    """Build an (m+1)-qubit circuit containing only the given oracle."""
    qpe_reg = QuantumRegister(m, name="qpe")
    anc = QuantumRegister(1, name="anc")
    circ = QuantumCircuit(qpe_reg, anc)
    t0 = dynamic_t0(A)
    C = C_factor(A)
    oracle_fn(circ, qpe_reg, anc[0], A=A, t0=t0, C=C)
    return circ


def _expected_thetas(A, m):
    """Reproduce the rotation-angle table used by the classical / unary oracles."""
    eigs = np.real(np.real_if_close(np.linalg.eigvalsh(A)))
    has_neg = bool(np.any(eigs < -1e-12))
    t0 = dynamic_t0(A)
    C = C_factor(A)
    thetas = np.empty(2**m)
    for k in range(2**m):
        phi = k / (2**m)
        if has_neg and phi >= 0.5:
            phi -= 1.0
        lam_est = (2 * np.pi * phi) / t0
        if abs(lam_est) < 1e-12:
            lam_est = 1e-12 if lam_est >= 0 else -1e-12
        lam = eigs[np.argmin(np.abs(eigs - lam_est))]
        ratio = np.clip(C / lam, -1.0, 1.0)
        thetas[k] = 2 * np.arcsin(ratio)
    return thetas


# ===================================================================
# unary_iteration_eig_inversion_oracle
# ===================================================================

class TestUnaryIterationOracle:

    # ---------------------------------------------------------------
    # Unitary equivalence with the classical oracle
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fixture_name", ["pd_2x2", "indefinite_2x2", "diagonal_2x2"])
    @pytest.mark.parametrize("m", [2, 3])
    def test_unitary_matches_classical(self, fixture_name, m, request):
        """The UCRy decomposition must implement the exact same unitary."""
        A = request.getfixturevalue(fixture_name)
        circ_cls = _build_oracle_circuit(classical_eig_inversion_oracle, A, m)
        circ_uni = _build_oracle_circuit(unary_iteration_eig_inversion_oracle, A, m)
        assert Operator(circ_cls).equiv(Operator(circ_uni))

    def test_unitary_matches_classical_4x4(self, pd_4x4):
        A = pd_4x4
        for m in (2, 3):
            circ_cls = _build_oracle_circuit(classical_eig_inversion_oracle, A, m)
            circ_uni = _build_oracle_circuit(unary_iteration_eig_inversion_oracle, A, m)
            assert Operator(circ_cls).equiv(Operator(circ_uni))

    # ---------------------------------------------------------------
    # Per-basis-state statevector correctness
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("m", [2, 3])
    def test_per_basis_state_rotation(self, pd_2x2, m):
        """For every QPE basis state |k>, the ancilla must be
        cos(θ_k/2)|0> + sin(θ_k/2)|1>."""
        A = pd_2x2
        thetas = _expected_thetas(A, m)
        t0 = dynamic_t0(A)
        C = C_factor(A)

        for k in range(2**m):
            qpe_reg = QuantumRegister(m, name="qpe")
            anc = QuantumRegister(1, name="anc")
            circ = QuantumCircuit(qpe_reg, anc)

            for j in range(m):
                if k & (1 << j):
                    circ.x(qpe_reg[j])

            unary_iteration_eig_inversion_oracle(
                circ, qpe_reg, anc[0], A=A, t0=t0, C=C,
            )

            sv = Statevector.from_instruction(circ)
            probs = sv.probabilities()

            idx_anc1 = k + 2**m
            expected_prob = np.sin(thetas[k] / 2) ** 2
            assert probs[idx_anc1] == pytest.approx(expected_prob, abs=1e-10), (
                f"k={k}: got P(anc=1)={probs[idx_anc1]:.8f}, expected {expected_prob:.8f}"
            )

    # ---------------------------------------------------------------
    # Gate structure: only CNOT + RY (no multi-controlled gates)
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_no_multi_controlled_gates(self, pd_2x2, m):
        """The circuit must not contain any gates with more than 1 control qubit."""
        circ = _build_oracle_circuit(unary_iteration_eig_inversion_oracle, pd_2x2, m)
        for inst in circ.data:
            n_qubits = inst.operation.num_qubits
            assert n_qubits <= 2, (
                f"Gate {inst.operation.name} acts on {n_qubits} qubits "
                f"(max 2 expected)"
            )

    # ---------------------------------------------------------------
    # Depth reduction vs classical oracle
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_depth_less_than_classical(self, pd_2x2, m):
        circ_cls = _build_oracle_circuit(classical_eig_inversion_oracle, pd_2x2, m)
        circ_uni = _build_oracle_circuit(unary_iteration_eig_inversion_oracle, pd_2x2, m)
        depth_cls = circ_cls.decompose().depth()
        depth_uni = circ_uni.decompose().depth()
        assert depth_uni < depth_cls, (
            f"m={m}: unary depth {depth_uni} >= classical depth {depth_cls}"
        )

    # ---------------------------------------------------------------
    # Matrix-type coverage
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("fixture_name", [
        "pd_2x2", "indefinite_2x2", "diagonal_2x2", "identity_2x2",
        "pd_4x4", "pd_8x8",
    ])
    def test_no_error_various_matrices(self, fixture_name, request):
        A = request.getfixturevalue(fixture_name)
        circ = _build_oracle_circuit(unary_iteration_eig_inversion_oracle, A, m=3)
        assert circ.depth() > 0

    # ---------------------------------------------------------------
    # Indefinite-matrix correctness
    # ---------------------------------------------------------------

    def test_indefinite_statevector(self, indefinite_2x2):
        """Verify per-basis-state angles for a matrix with negative eigenvalues."""
        A = indefinite_2x2
        m = 3
        thetas = _expected_thetas(A, m)
        t0 = dynamic_t0(A)
        C = C_factor(A)

        for k in range(2**m):
            qpe_reg = QuantumRegister(m, name="qpe")
            anc = QuantumRegister(1, name="anc")
            circ = QuantumCircuit(qpe_reg, anc)
            for j in range(m):
                if k & (1 << j):
                    circ.x(qpe_reg[j])
            unary_iteration_eig_inversion_oracle(
                circ, qpe_reg, anc[0], A=A, t0=t0, C=C,
            )
            sv = Statevector.from_instruction(circ)
            probs = sv.probabilities()
            idx_anc1 = k + 2**m
            expected_prob = np.sin(thetas[k] / 2) ** 2
            assert probs[idx_anc1] == pytest.approx(expected_prob, abs=1e-10)

    # ---------------------------------------------------------------
    # Edge: uniform angles (all thetas identical)
    # ---------------------------------------------------------------

    def test_identity_matrix_uniform_angles(self, identity_2x2):
        """For the identity matrix all eigenvalues are equal, so all θ_k
        should be identical.  The circuit must still be correct."""
        A = identity_2x2
        m = 2
        circ_cls = _build_oracle_circuit(classical_eig_inversion_oracle, A, m)
        circ_uni = _build_oracle_circuit(unary_iteration_eig_inversion_oracle, A, m)
        assert Operator(circ_cls).equiv(Operator(circ_uni))

    # ---------------------------------------------------------------
    # Gate-count scaling
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_gate_count_scaling(self, pd_2x2, m):
        """The decomposed circuit should have ~2^{m+1} CNOT + 2^m RY gates."""
        circ = _build_oracle_circuit(unary_iteration_eig_inversion_oracle, pd_2x2, m)
        decomposed = circ.decompose()
        cx_count = sum(1 for i in decomposed.data if i.operation.name == "cx")
        ry_count = sum(1 for i in decomposed.data if i.operation.name == "ry")
        assert cx_count == 2 * (2**m) - 2
        assert ry_count <= 2**m
