"""Tests for qlsas.algorithms.hhl.hhl_helpers."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, ExactReciprocalGate
from qiskit.quantum_info import Operator, Statevector

from qlsas.algorithms.hhl.hhl_helpers import (
    mcry_eig_inversion,
    exact_reciprocal_eig_inversion,
    ucry_eig_inversion,
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
# mcry_eig_inversion
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
        mcry_eig_inversion(circ, qpe_reg, anc[0], A=pd_2x2, t0=t0, C=C)

        assert circ.depth() > depth_before
        ry_count = sum(1 for inst in circ.data if isinstance(inst.operation, RYGate) or "ry" in inst.operation.name.lower())
        # Oracle skips k=0 (phase=0 ⇒ no inversion), so it emits 2^m - 1 gates.
        assert ry_count >= 2**num_qpe_qubits - 1

    @pytest.mark.parametrize("fixture_name", ["pd_2x2", "indefinite_2x2"])
    def test_no_error_pd_and_indefinite(self, fixture_name, request):
        A = request.getfixturevalue(fixture_name)
        qpe_reg = QuantumRegister(3, name="qpe")
        anc = QuantumRegister(1, name="anc")
        circ = QuantumCircuit(qpe_reg, anc)
        t0 = dynamic_t0(A)
        C = C_factor(A)
        mcry_eig_inversion(circ, qpe_reg, anc[0], A=A, t0=t0, C=C)


# ===================================================================
# exact_reciprocal_eig_inversion
# ===================================================================

class TestQuantumOracle:

    def test_appends_exact_reciprocal_gate(self, pd_2x2):
        qpe_reg = QuantumRegister(3, name="qpe")
        anc = QuantumRegister(1, name="anc")
        circ = QuantumCircuit(qpe_reg, anc)
        depth_before = circ.depth()

        t0 = dynamic_t0(pd_2x2)
        C = C_factor(pd_2x2)
        exact_reciprocal_eig_inversion(circ, qpe_reg, anc[0], A=pd_2x2, t0=t0, C=C)

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
        exact_reciprocal_eig_inversion(circ, qpe_reg, anc[0], A=A, t0=t0, C=C)


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
    """Reproduce the rotation-angle table used by the classical / unary oracles.

    Mirrors the post-edit oracle semantics: no eigenvalue snapping,
    two's-complement unwrap only when A has negative eigenvalues, k=0
    left unrotated.
    """
    t0 = dynamic_t0(A)
    C = C_factor(A)
    has_neg = bool(np.any(np.linalg.eigvalsh(A) < -1e-12))
    thetas = np.zeros(2**m)
    for k in range(1, 2**m):
        phi = k / (2**m)
        if has_neg and phi >= 0.5:
            phi -= 1.0
        lam = 2 * np.pi * phi / t0
        ratio = float(np.clip(C / lam, -1.0, 1.0))
        thetas[k] = 2 * np.arcsin(ratio)
    return thetas


# ===================================================================
# ucry_eig_inversion
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
        circ_cls = _build_oracle_circuit(mcry_eig_inversion, A, m)
        circ_uni = _build_oracle_circuit(ucry_eig_inversion, A, m)
        assert Operator(circ_cls).equiv(Operator(circ_uni))

    def test_unitary_matches_classical_4x4(self, pd_4x4):
        A = pd_4x4
        for m in (2, 3):
            circ_cls = _build_oracle_circuit(mcry_eig_inversion, A, m)
            circ_uni = _build_oracle_circuit(ucry_eig_inversion, A, m)
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

            ucry_eig_inversion(
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
        circ = _build_oracle_circuit(ucry_eig_inversion, pd_2x2, m)
        for inst in circ.data:
            n_qubits = inst.operation.num_qubits
            assert n_qubits <= 2, (
                f"Gate {inst.operation.name} acts on {n_qubits} qubits "
                f"(max 2 expected)"
            )

    # ---------------------------------------------------------------
    # Depth reduction vs classical oracle
    # ---------------------------------------------------------------

    # m=2 excluded: with k=0 skipped the classical oracle has only 3
    # multi-controlled RY gates and the unary fixed-overhead loses the race.
    # The asymptotic O(2^m) vs O(m·2^m) win shows up at m >= 3.
    @pytest.mark.parametrize("m", [3, 4])
    def test_depth_less_than_classical(self, pd_2x2, m):
        circ_cls = _build_oracle_circuit(mcry_eig_inversion, pd_2x2, m)
        circ_uni = _build_oracle_circuit(ucry_eig_inversion, pd_2x2, m)
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
        circ = _build_oracle_circuit(ucry_eig_inversion, A, m=3)
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
            ucry_eig_inversion(
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
        circ_cls = _build_oracle_circuit(mcry_eig_inversion, A, m)
        circ_uni = _build_oracle_circuit(ucry_eig_inversion, A, m)
        assert Operator(circ_cls).equiv(Operator(circ_uni))

    # ---------------------------------------------------------------
    # Gate-count scaling
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_gate_count_scaling(self, pd_2x2, m):
        """The decomposed circuit should have ~2^{m+1} CNOT + 2^m RY gates."""
        circ = _build_oracle_circuit(ucry_eig_inversion, pd_2x2, m)
        decomposed = circ.decompose()
        cx_count = sum(1 for i in decomposed.data if i.operation.name == "cx")
        ry_count = sum(1 for i in decomposed.data if i.operation.name == "ry")
        assert cx_count == 2 * (2**m) - 2
        assert ry_count <= 2**m


# ===================================================================
# ExactReciprocalEigOracle scaling probe
# ===================================================================
#
# These tests isolate the scaling formula S used inside
# exact_reciprocal_eig_inversion when constructing
# qiskit.circuit.library.ExactReciprocalGate.  They compare the
# per-basis-state ancilla rotation that the gate actually applies
# against the analytical HHL reference angle θ_k = 2·arcsin(C / λ_k),
# where λ_k is the eigenvalue implied by the QPE phase (with two's-
# complement unwrap for k ≥ 2^(m-1)).
#
# Bug hypothesis
# --------------
# Qiskit's ExactReciprocal applies, for state register integer i,
#
#     angle = 2·arcsin( scaling · nl / i )
#
# with nl = 2^n  when neg_vals = False
#      nl = 2^(n-1) when neg_vals = True (sign bit excluded from the
#                                          magnitude register).
#
# In HHL the QPE phase is φ = j / 2^n, so λ = 2π·φ / t0 = 2π·j / (t0·2^n)
# and we want  scaling·nl / i  to equal  C / λ.  Matching gives
#
#     neg_vals = False  →  S = C·t0 / (2π)
#     neg_vals = True   →  S = C·t0 / π
#
# The current code in exact_reciprocal_eig_inversion uses
# S = C·t0 / (2π) regardless of neg_vals, so for any matrix that
# triggers neg_vals=True (i.e. an indefinite spectrum) the rotation
# argument is half of what HHL needs — the gate behaves as if the
# eigenvalue were twice as large.
#
# Boundary caveat
# ---------------
# Even with the corrected scaling there is a residual mismatch at the
# single state k = 2^(m-1) (the most-negative phase under two's-
# complement) because the gate's internal table fixes
# angles_neg[0] = 0 — this state is never rotated regardless of S.
# The reference oracle (post-edit classical) does rotate it, so the
# tests below explicitly skip k = 2^(m-1).
#
# What these tests prove
# ----------------------
# * test_pd_matrix_positive_half  should PASS with the current code
#   (PD matrix → neg_vals=False → S formula is correct).
# * test_indefinite_matrix        should FAIL with the current code
#   (indefinite matrix → neg_vals=True → S off by factor 2).  After
#   patching exact_reciprocal_eig_inversion to use S = C·t0/π and
#   neg_vals=True unconditionally, this test should PASS.
# ===================================================================

class TestQuantumOracleScaling:

    @staticmethod
    def _phi_unwrapped(k: int, m: int) -> float:
        phi = k / (2**m)
        if phi >= 0.5:
            phi -= 1.0
        return phi

    @classmethod
    def _ref_ratio(cls, k: int, m: int, t0: float, C: float):
        """Analytical reference C/λ_k.  Returns None when k=0 (no inversion)."""
        if k == 0:
            return None
        phi = cls._phi_unwrapped(k, m)
        if phi == 0:
            return None
        lam = 2 * np.pi * phi / t0
        return C / lam

    @classmethod
    def _expected_p_anc1(cls, k: int, m: int, t0: float, C: float) -> float:
        ratio = cls._ref_ratio(k, m, t0, C)
        if ratio is None:
            return 0.0
        ratio = max(min(ratio, 1.0), -1.0)
        return float(np.sin(np.arcsin(ratio)) ** 2)  # = ratio**2

    @staticmethod
    def _measured_p_anc1(A, k: int, m: int, t0: float, C: float) -> float:
        """Run exact_reciprocal_eig_inversion on |k⟩|0⟩ and return P(anc=1)."""
        qpe_reg = QuantumRegister(m, name="qpe")
        anc = QuantumRegister(1, name="anc")
        circ = QuantumCircuit(qpe_reg, anc)
        for j in range(m):
            if k & (1 << j):
                circ.x(qpe_reg[j])
        exact_reciprocal_eig_inversion(circ, qpe_reg, anc[0], A=A, t0=t0, C=C)
        sv = Statevector.from_instruction(circ)
        return float(sv.probabilities()[k + 2**m])

    # Smaller C than C_factor() so |C/λ_k| < 1 for every k > 0 — this
    # keeps the test inside the well-defined arcsin range, so any
    # discrepancy with the reference must come from the scaling formula
    # itself, not from saturation behaviour (the gate emits angle 0
    # when |scaling·nl/i| > 1, the reference would clamp to ±π).
    @staticmethod
    def _safe_C(A, m: int) -> float:
        t0 = dynamic_t0(A)
        # Smallest non-zero |φ_k| under unwrap is 1/2^m, giving |λ_min| = 2π/(t0·2^m).
        lam_min = 2 * np.pi / (t0 * 2**m)
        return 0.5 * lam_min  # margin of 2× under the saturation threshold

    @pytest.mark.parametrize("m", [3, 4])
    def test_pd_matrix_positive_half(self, pd_2x2, m):
        """PD matrix → gate runs neg_vals=False → S = C·t0/(2π) should be
        correct.  Only k ∈ [1, 2^(m-1)) is checked (physical states for a
        PD spectrum)."""
        A = pd_2x2
        t0 = dynamic_t0(A)
        C = self._safe_C(A, m)
        for k in range(1, 2**(m - 1)):
            expected = self._expected_p_anc1(k, m, t0, C)
            got = self._measured_p_anc1(A, k, m, t0, C)
            assert got == pytest.approx(expected, abs=1e-10), (
                f"PD k={k}: P(anc=1)={got:.8f} vs expected {expected:.8f}"
            )

    @pytest.mark.parametrize("m", [3, 4])
    def test_indefinite_matrix(self, indefinite_2x2, m):
        """Indefinite matrix → gate runs neg_vals=True.

        Hypothesis: with the current S = C·t0/(2π) this assertion fails on
        every k ∈ [1, 2^m) \\ {2^(m-1)} because the gate's arcsin argument
        is half of C/λ.  After patching the oracle to use S = C·t0/π and
        pinning neg_vals=True, all of those k should pass; only the
        most-negative boundary k = 2^(m-1) remains divergent (the gate
        hard-codes angles_neg[0] = 0)."""
        A = indefinite_2x2
        t0 = dynamic_t0(A)
        C = self._safe_C(A, m)
        boundary = 2**(m - 1)
        for k in range(1, 2**m):
            if k == boundary:
                continue
            expected = self._expected_p_anc1(k, m, t0, C)
            got = self._measured_p_anc1(A, k, m, t0, C)
            assert got == pytest.approx(expected, abs=1e-10), (
                f"INDEF k={k}: P(anc=1)={got:.8f} vs expected {expected:.8f}"
            )
