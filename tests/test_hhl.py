"""Tests for qlsas.algorithms.hhl.hhl.HHL."""

import math
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qlsas.state_prep import StatePrep
from qlsas.algorithms.hhl import HHL, MCRYEigOracle, ExactReciprocalEigOracle, UCRYEigOracle
from qlsas.readout.base import QLSACircuit


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
# Circuit structure — core HHL (no readout)
# ===================================================================

class TestCoreCircuit:

    @pytest.mark.parametrize("fixture_A, fixture_b, oracle", [
        ("pd_2x2", "b_2", MCRYEigOracle()),
        ("pd_2x2", "b_2", ExactReciprocalEigOracle()),
        ("pd_4x4", "b_4", MCRYEigOracle()),
        ("indefinite_2x2", "b_2", MCRYEigOracle()),
    ])
    def test_register_names(self, fixture_A, fixture_b, oracle, state_prep, request):
        A = request.getfixturevalue(fixture_A)
        b = request.getfixturevalue(fixture_b)
        hhl = HHL(num_qpe_qubits=4, eig_oracle=oracle)
        result = hhl.build_circuit(A, b, state_prep)

        assert isinstance(result, QLSACircuit)
        names = _register_names(result.circuit)
        assert "ancilla_flag_register" in names
        assert "qpe_register" in names
        assert "b_to_x_register" in names
        assert "ancilla_flag_result" in names

    @pytest.mark.parametrize("num_qpe, size, fixture_A, fixture_b", [
        (3, 2, "pd_2x2", "b_2"),
        (4, 2, "pd_2x2", "b_2"),
        (4, 4, "pd_4x4", "b_4"),
    ])
    def test_qubit_count(self, num_qpe, size, fixture_A, fixture_b, state_prep, request):
        A = request.getfixturevalue(fixture_A)
        b = request.getfixturevalue(fixture_b)
        hhl = HHL(num_qpe_qubits=num_qpe, eig_oracle=MCRYEigOracle())
        result = hhl.build_circuit(A, b, state_prep)
        expected = 1 + num_qpe + int(math.log2(size))
        assert result.circuit.num_qubits == expected

    @pytest.mark.slow
    def test_8x8_circuit_builds(self, pd_8x8, b_8, state_prep):
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        result = hhl.build_circuit(pd_8x8, b_8, state_prep)
        expected = 1 + 4 + 3  # ancilla + qpe + log2(8)
        assert result.circuit.num_qubits == expected

    def test_circuit_name(self, pd_2x2, b_2, state_prep):
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        result = hhl.build_circuit(pd_2x2, b_2, state_prep)
        assert "HHL" in result.circuit.name
        assert "2 by 2" in result.circuit.name

    def test_qlsa_circuit_registers(self, pd_2x2, b_2, state_prep):
        """Verify the QLSACircuit metadata fields are populated correctly."""
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        result = hhl.build_circuit(pd_2x2, b_2, state_prep)
        assert result.solution_register.name == "b_to_x_register"
        assert result.ancilla_register.name == "ancilla_flag_register"
        assert result.ancilla_creg.name == "ancilla_flag_result"
        assert len(result.solution_register) == 1  # log2(2) = 1
        assert len(result.ancilla_register) == 1

    def test_qlsa_circuit_params(self, pd_2x2, b_2, state_prep):
        """QLSACircuit.params should contain t0 and C."""
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        result = hhl.build_circuit(pd_2x2, b_2, state_prep)
        assert "t0" in result.params
        assert "C" in result.params
        assert isinstance(result.params["t0"], float)
        assert isinstance(result.params["C"], float)

    def test_explicit_t0_and_C_stored_in_params(self, pd_2x2, b_2, state_prep):
        """Explicit t0 / C kwargs should appear verbatim in QLSACircuit.params."""
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        result = hhl.build_circuit(pd_2x2, b_2, state_prep, t0=0.5, C=0.1)
        assert result.params["t0"] == 0.5
        assert result.params["C"] == 0.1


# ===================================================================
# Input validation
# ===================================================================

class TestBuildCircuitValidation:

    def test_non_square_A(self, b_2, state_prep):
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        A = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="square"):
            hhl.build_circuit(A, b_2, state_prep)

    def test_non_vector_b(self, pd_2x2, state_prep):
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="vector"):
            hhl.build_circuit(pd_2x2, b, state_prep)

    def test_dimension_mismatch(self, pd_4x4, b_2, state_prep):
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        with pytest.raises(ValueError, match="mismatch"):
            hhl.build_circuit(pd_4x4, b_2, state_prep)

    def test_non_hermitian(self, b_2, state_prep):
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        A = np.array([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="Hermitian"):
            hhl.build_circuit(A, b_2, state_prep)

    def test_non_unit_norm_b(self, pd_2x2, state_prep):
        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        b = np.array([3.0, 4.0])
        with pytest.raises(ValueError, match="unit norm"):
            hhl.build_circuit(pd_2x2, b, state_prep)

    def test_invalid_eig_oracle_type_raises(self):
        """Passing a non-EigOracle raises TypeError at construction time."""
        with pytest.raises(TypeError, match="EigOracle"):
            HHL(num_qpe_qubits=4, eig_oracle="classical")  # type: ignore[arg-type]

    def test_default_oracle_is_classical(self, pd_2x2, b_2, state_prep):
        """HHL() without eig_oracle defaults to MCRYEigOracle."""
        hhl = HHL(num_qpe_qubits=4)
        result = hhl.build_circuit(pd_2x2, b_2, state_prep)
        assert isinstance(result, QLSACircuit)


# ===================================================================
# Parameterized across QPE qubits and oracles
# ===================================================================

@pytest.mark.parametrize("num_qpe", [3, 4])
@pytest.mark.parametrize("oracle", [MCRYEigOracle(), ExactReciprocalEigOracle()])
def test_builds_for_various_configs(pd_2x2, b_2, state_prep, num_qpe, oracle):
    hhl = HHL(num_qpe_qubits=num_qpe, eig_oracle=oracle)
    result = hhl.build_circuit(pd_2x2, b_2, state_prep)
    assert isinstance(result, QLSACircuit)
    assert isinstance(result.circuit, QuantumCircuit)
    assert result.circuit.num_qubits == 1 + num_qpe + 1  # ancilla + qpe + 1 data qubit for 2x2


# ===================================================================
# Caller-supplied spectral information avoids classical eigendecomposition
# ===================================================================

class TestNoClassicalEigendecomposition:
    """The user can avoid every internal np.linalg.eigvalsh(A) call by
    supplying t0 (or lambda_max_bound), C (or lambda_min_bound), and
    has_negative_eigenvalues."""

    @pytest.mark.parametrize("oracle", [
        MCRYEigOracle(),
        UCRYEigOracle(),
        ExactReciprocalEigOracle(),
    ])
    def test_explicit_t0_C_and_sign_skip_eigvalsh(self, pd_2x2, b_2, state_prep, oracle):
        hhl = HHL(num_qpe_qubits=3, eig_oracle=oracle)
        with patch("numpy.linalg.eigvalsh") as mock_eigvalsh:
            hhl.build_circuit(
                pd_2x2,
                b_2,
                state_prep,
                t0=0.5,
                C=0.1,
                has_negative_eigenvalues=False,
            )
        mock_eigvalsh.assert_not_called()

    def test_bounds_replace_eigvalsh(self, pd_2x2, b_2, state_prep):
        """Supplying lambda_*_bound (not t0/C) also avoids eigvalsh."""
        hhl = HHL(num_qpe_qubits=3)
        with patch("numpy.linalg.eigvalsh") as mock_eigvalsh:
            result = hhl.build_circuit(
                pd_2x2,
                b_2,
                state_prep,
                lambda_max_bound=4.0,
                lambda_min_bound=0.5,
                has_negative_eigenvalues=False,
            )
        mock_eigvalsh.assert_not_called()
        # Resolution rules: t0 = (pi/lambda_max)*(1-0.05); C = 0.9*lambda_min
        assert result.params["t0"] == pytest.approx((np.pi / 4.0) * 0.95)
        assert result.params["C"] == pytest.approx(0.45)

    def test_default_path_still_calls_eigvalsh(self, pd_2x2, b_2, state_prep):
        """Backward compatibility: omitting all three kwargs preserves the
        legacy auto-detect path."""
        hhl = HHL(num_qpe_qubits=3, eig_oracle=MCRYEigOracle())
        with patch("numpy.linalg.eigvalsh", wraps=np.linalg.eigvalsh) as mock_eigvalsh:
            hhl.build_circuit(pd_2x2, b_2, state_prep)
        assert mock_eigvalsh.call_count >= 1

    def test_explicit_sign_for_indefinite(self, indefinite_2x2, b_2, state_prep):
        """has_negative_eigenvalues=True is honoured for indefinite A."""
        hhl = HHL(num_qpe_qubits=3, eig_oracle=UCRYEigOracle())
        with patch("numpy.linalg.eigvalsh") as mock_eigvalsh:
            hhl.build_circuit(
                indefinite_2x2,
                b_2,
                state_prep,
                t0=0.5,
                C=0.1,
                has_negative_eigenvalues=True,
            )
        mock_eigvalsh.assert_not_called()
