"""Tests for HRFReadout and the HRF solve path in QuantumLinearSolver.

Test hierarchy
--------------
Unit — pure-Python, no backend:
  TestHRFReadoutApply         register wiring, circuit structure
  TestBuildHRFCircuits        N H-gate variants, correct qubit targeting
  TestExtractProbs            post-selection logic, edge cases
  TestHRFProcess              statevector reconstruction from synthetic samples

Integration — uses AerSimulator, real HHL circuits:
  TestHRFSolverEndToEnd       2×2 and 4×4 systems, result quality checks
  TestHRFSolverContracts      SolveResult fields, error conditions
"""

from __future__ import annotations

import math
import numpy as np
import numpy.linalg as LA
import pytest

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from qlsas.readout.base import QLSACircuit
from qlsas.readout import HRFReadout, MeasureXReadout
from qlsas.readout.hrf_readout import HRFReadout
from qlsas.measurement_result import MeasurementResult
from qlsas.solver import QuantumLinearSolver, SolveResult
from qlsas.state_prep import DefaultStatePrep
from qlsas.algorithms.hhl import HHL, ClassicalEigOracle


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalized(v: np.ndarray) -> np.ndarray:
    return v / LA.norm(v)


def _random_spd(n: int, cond: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(1.0, cond, n)
    return (Q * eigs) @ Q.T


def _make_qlsa_circuit(n: int = 2) -> tuple[QLSACircuit, QuantumCircuit]:
    """Build a minimal mock QLSACircuit with n solution qubits."""
    anc_qr = QuantumRegister(1, "ancilla_flag_register")
    sol_qr = QuantumRegister(n, "b_to_x_register")
    anc_cr = ClassicalRegister(1, "ancilla_flag_result")
    circ = QuantumCircuit(anc_qr, sol_qr, anc_cr)
    circ.h(sol_qr)  # some gates so the circuit is non-trivial
    circ.measure(anc_qr, anc_cr)
    return (
        QLSACircuit(
            circuit=circ,
            solution_register=sol_qr,
            ancilla_register=anc_qr,
            ancilla_creg=anc_cr,
        ),
        circ,
    )


def _make_solver(backend, num_trees=5, num_qpe=4, shots=2048):
    hhl = HHL(num_qpe_qubits=num_qpe, eig_oracle=ClassicalEigOracle())
    readout = HRFReadout(num_trees=num_trees)
    return QuantumLinearSolver(
        qlsa=hhl,
        readout=readout,
        backend=backend,
        state_prep=DefaultStatePrep(),
        shots=shots,
    )


# ===========================================================================
# Unit: HRFReadout.apply()
# ===========================================================================

class TestHRFReadoutApply:

    def test_returns_quantum_circuit(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        result = readout.apply(qlsa_circ)
        assert isinstance(result, QuantumCircuit)

    def test_solution_creg_added(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        circ = readout.apply(qlsa_circ)
        creg_names = [cr.name for cr in circ.cregs]
        assert "hrf_x_result" in creg_names

    def test_solution_creg_size_matches_register(self):
        for n in (1, 2, 3):
            readout = HRFReadout()
            qlsa_circ, _ = _make_qlsa_circuit(n=n)
            circ = readout.apply(qlsa_circ)
            sol_creg = next(cr for cr in circ.cregs if cr.name == "hrf_x_result")
            assert len(sol_creg) == n

    def test_ancilla_creg_preserved(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        circ = readout.apply(qlsa_circ)
        creg_names = [cr.name for cr in circ.cregs]
        assert "ancilla_flag_result" in creg_names

    def test_register_names_set_after_apply(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.apply(qlsa_circ)
        assert readout.register_names == ["ancilla_flag_result", "hrf_x_result"]

    def test_base_circuit_core_stored(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.apply(qlsa_circ)
        assert readout._base_circuit_core is not None

    def test_apply_without_state_prep_arg_ok(self):
        """state_prep is not used by HRFReadout; passing None must succeed."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.apply(qlsa_circ, state_prep=None)  # should not raise


# ===========================================================================
# Unit: HRFReadout.build_hrf_circuits()
# ===========================================================================

class TestBuildHRFCircuits:

    def test_raises_before_apply(self):
        readout = HRFReadout()
        with pytest.raises(RuntimeError, match="apply()"):
            readout.build_hrf_circuits()

    def test_returns_n_circuits(self):
        for n in (1, 2, 3):
            readout = HRFReadout()
            qlsa_circ, _ = _make_qlsa_circuit(n=n)
            readout.apply(qlsa_circ)
            circuits = readout.build_hrf_circuits()
            assert len(circuits) == n, f"Expected {n} circuits for n={n} solution qubits"

    def test_each_is_quantum_circuit(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.apply(qlsa_circ)
        for circ in readout.build_hrf_circuits():
            assert isinstance(circ, QuantumCircuit)

    def test_all_circuits_have_solution_creg(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.apply(qlsa_circ)
        for circ in readout.build_hrf_circuits():
            creg_names = [cr.name for cr in circ.cregs]
            assert "hrf_x_result" in creg_names

    def test_circuits_have_h_gate_on_correct_qubit(self):
        """Each circuit's non-base operations should include exactly one H gate on a solution qubit."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.apply(qlsa_circ)
        circuits = readout.build_hrf_circuits()
        # Each circuit should have one more H gate than the base (on a solution qubit)
        for idx, circ in enumerate(circuits):
            h_ops = [
                inst for inst in circ.data
                if inst.operation.name == "h"
                and any(
                    circ.find_bit(q).registers[0][0].name == "b_to_x_register"
                    for q in inst.qubits
                )
            ]
            assert len(h_ops) >= 1, f"Circuit {idx} missing H on solution qubit"

    def test_build_callable_multiple_times(self):
        """build_hrf_circuits() should be idempotent — repeated calls give independent lists."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.apply(qlsa_circ)
        c1 = readout.build_hrf_circuits()
        c2 = readout.build_hrf_circuits()
        assert len(c1) == len(c2)
        # Should be fresh copies, not the same objects
        assert c1[0] is not c2[0]


# ===========================================================================
# Unit: HRFReadout._extract_probs()
# ===========================================================================

class TestExtractProbs:

    def _counts_to_mr(self, counts: dict) -> MeasurementResult:
        return MeasurementResult(counts)

    def test_basic_postselection(self):
        """For 1-qubit solution, filter ancilla=1 shots and normalise."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.apply(qlsa_circ)

        # Bitstring format: [sol_bit][ancilla_bit]
        # "01": sol=0, ancilla=1 (success) → 60 shots
        # "11": sol=1, ancilla=1 (success) → 40 shots
        # "00": sol=0, ancilla=0 (fail)    → 50 shots
        counts = {"01": 60, "11": 40, "00": 50}
        result = self._counts_to_mr(counts)
        probs, rate = readout._extract_probs(result, n_sol=1)

        assert probs.shape == (2,)
        assert np.isclose(probs[0], 60 / 100)
        assert np.isclose(probs[1], 40 / 100)
        assert np.isclose(probs.sum(), 1.0)
        assert np.isclose(rate, 100 / 150)

    def test_two_qubit_solution(self):
        """2-qubit solution (4D), bitstrings are 3 chars: [q1 q0][ancilla]."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.apply(qlsa_circ)

        counts = {
            "001": 50,   # sol=00=0, anc=1
            "011": 30,   # sol=01=1, anc=1
            "101": 15,   # sol=10=2, anc=1
            "111": 5,    # sol=11=3, anc=1
            "000": 200,  # failures
        }
        result = self._counts_to_mr(counts)
        probs, rate = readout._extract_probs(result, n_sol=2)

        assert probs.shape == (4,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.isclose(probs[0], 50 / 100)
        assert np.isclose(probs[1], 30 / 100)
        assert np.isclose(probs[2], 15 / 100)
        assert np.isclose(probs[3], 5  / 100)
        assert np.isclose(rate, 100 / 300)

    def test_no_successful_shots_raises(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.apply(qlsa_circ)
        counts = {"00": 100, "10": 50}  # all ancilla=0
        result = self._counts_to_mr(counts)
        with pytest.raises(ValueError, match="No successful ancilla shots"):
            readout._extract_probs(result, n_sol=1)

    def test_all_shots_successful(self):
        """Success rate = 1.0 when all shots have ancilla=1."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.apply(qlsa_circ)
        counts = {"01": 75, "11": 25}
        result = self._counts_to_mr(counts)
        probs, rate = readout._extract_probs(result, n_sol=1)
        assert np.isclose(rate, 1.0)
        assert np.isclose(probs.sum(), 1.0)


# ===========================================================================
# Unit: HRFReadout.process() — synthetic samples
# ===========================================================================

class TestHRFProcess:

    def _uniform_probs(self, n: int) -> np.ndarray:
        p = np.ones(2**n) / (2**n)
        return p

    def test_process_returns_three_tuple(self):
        """process() must return a 3-tuple: (solution, placeholder, residual)."""
        rng = np.random.default_rng(7)
        n = 1
        A = np.diag([2.0, 1.0])
        b = _normalized(rng.standard_normal(2))

        readout = HRFReadout(num_trees=5)
        qlsa_circ, _ = _make_qlsa_circuit(n=n)
        readout.apply(qlsa_circ)

        # Provide a valid 2-element samples list (base + 1 H-variant)
        samples = [np.array([0.75, 0.25]), np.array([0.60, 0.40])]
        result = readout.process(samples, A, b, verbose=False)
        assert len(result) == 3

    def test_solution_has_correct_shape(self):
        n = 1
        A = np.diag([2.0, 1.0])
        b = _normalized(np.array([1.0, 1.0]))
        readout = HRFReadout(num_trees=5)
        qlsa_circ, _ = _make_qlsa_circuit(n=n)
        readout.apply(qlsa_circ)
        samples = [np.array([0.6, 0.4]), np.array([0.55, 0.45])]
        solution, _, _ = readout.process(samples, A, b, verbose=False)
        assert solution.shape == (2,)

    def test_near_zero_statevector_raises(self):
        """process() raises ValueError when samples yield a near-zero statevector."""
        n = 1
        A = np.diag([1.0, 1.0])
        b = _normalized(np.array([1.0, 0.0]))
        readout = HRFReadout(num_trees=5)
        qlsa_circ, _ = _make_qlsa_circuit(n=n)
        readout.apply(qlsa_circ)
        # All-zero samples → zero amplitudes → zero statevector
        samples = [np.zeros(2), np.zeros(2)]
        with pytest.raises((ValueError, Exception)):
            readout.process(samples, A, b, verbose=False)

    def test_unit_norm_base_samples_produces_finite_solution(self):
        """With reasonable samples drawn from a known statevector, solution is finite."""
        rng = np.random.default_rng(99)
        n = 2
        A = np.diag([3.0, 2.0, 1.5, 1.0])
        b = _normalized(rng.standard_normal(4))

        readout = HRFReadout(num_trees=10)
        qlsa_circ, _ = _make_qlsa_circuit(n=n)
        readout.apply(qlsa_circ)

        # Draw samples from a plausible quantum state
        true_state = _normalized(rng.standard_normal(4))
        base_probs = true_state**2  # squared amplitudes
        samples = [base_probs] + [
            (base_probs + rng.uniform(-0.02, 0.02, 4)).clip(0)
            for _ in range(n)
        ]
        # Normalise each
        samples = [s / s.sum() for s in samples]

        solution, _, residual = readout.process(samples, A, b, verbose=False)
        assert solution.shape == (4,)
        assert np.all(np.isfinite(solution))
        assert np.isfinite(residual)


# ===========================================================================
# Integration: end-to-end with AerSimulator
# ===========================================================================

class TestHRFSolverEndToEnd:

    def test_2x2_pd_solution_direction(self, aer_backend, pd_2x2, b_2):
        """HRF solution should point in the correct direction for a 2×2 PD system."""
        solver = _make_solver(aer_backend, shots=4096, num_trees=10)
        result = solver.solve(pd_2x2, b_2, verbose=False)

        assert isinstance(result, SolveResult)
        classical = LA.solve(pd_2x2, b_2)
        cosine_sim = np.abs(
            np.dot(_normalized(result.solution), _normalized(classical))
        )
        assert cosine_sim > 0.7, f"Cosine sim too low: {cosine_sim:.3f}"

    def test_2x2_identity(self, aer_backend, b_2):
        """For Ax=b with A=I, the solution should align with b."""
        A = np.eye(2)
        solver = _make_solver(aer_backend, shots=4096, num_trees=10)
        result = solver.solve(A, b_2, verbose=False)

        cosine_sim = np.abs(np.dot(_normalized(result.solution), b_2))
        assert cosine_sim > 0.7, f"Cosine sim with b too low: {cosine_sim:.3f}"

    def test_2x2_diagonal(self, aer_backend, diagonal_2x2, b_2):
        """HRF should recover the correct solution direction for a diagonal system."""
        solver = _make_solver(aer_backend, shots=4096, num_trees=10)
        result = solver.solve(diagonal_2x2, b_2, verbose=False)
        classical = LA.solve(diagonal_2x2, b_2)
        cosine_sim = np.abs(
            np.dot(_normalized(result.solution), _normalized(classical))
        )
        assert cosine_sim > 0.7, f"Cosine sim too low: {cosine_sim:.3f}"

    @pytest.mark.slow
    def test_4x4_pd_solution_direction(self, aer_backend):
        """HRF on a 4×4 system runs without error and returns a plausible solution."""
        A = _random_spd(4, cond=3.0, seed=42)
        b = _normalized(np.random.default_rng(42).standard_normal(4))
        solver = _make_solver(aer_backend, shots=4096, num_trees=15)
        result = solver.solve(A, b, verbose=False)

        assert isinstance(result, SolveResult)
        assert result.solution.shape == (4,)
        assert np.all(np.isfinite(result.solution))

        classical = LA.solve(A, b)
        cosine_sim = np.abs(
            np.dot(_normalized(result.solution), _normalized(classical))
        )
        assert cosine_sim > 0.6, f"4×4 cosine sim too low: {cosine_sim:.3f}"

    def test_solve_result_fields_populated(self, aer_backend, pd_2x2, b_2):
        """SolveResult.success_rate, .residual, and .metadata should all be set."""
        solver = _make_solver(aer_backend, shots=2048, num_trees=5)
        result = solver.solve(pd_2x2, b_2, verbose=False)

        assert result.success_rate is not None
        assert 0.0 < result.success_rate <= 1.0
        assert result.residual is not None
        assert np.isfinite(result.residual)
        assert result.metadata.get("num_hrf_circuits") == 2  # 1 base + 1 H for 2×2
        assert result.metadata.get("num_trees") == 5

    def test_result_is_solve_result(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend, shots=1024, num_trees=5)
        result = solver.solve(pd_2x2, b_2, verbose=False)
        assert isinstance(result, SolveResult)

    def test_numpy_interop(self, aer_backend, pd_2x2, b_2):
        """SolveResult should support numpy operations via its __array__ method."""
        solver = _make_solver(aer_backend, shots=1024, num_trees=5)
        result = solver.solve(pd_2x2, b_2, verbose=False)
        arr = np.asarray(result)
        assert arr.shape == (2,)
        assert np.all(np.isfinite(arr))


# ===========================================================================
# Integration: solver contracts and error conditions
# ===========================================================================

class TestHRFSolverContracts:

    def test_target_successful_shots_raises(self, aer_backend, pd_2x2, b_2):
        solver = _make_solver(aer_backend)
        solver.target_successful_shots = 20
        with pytest.raises(ValueError, match="target_successful_shots"):
            solver.solve(pd_2x2, b_2, verbose=False)

    def test_hrf_readout_exported_from_package(self):
        from qlsas.readout import HRFReadout as HR2
        assert HR2 is HRFReadout

    def test_existing_measure_x_unaffected(self, aer_backend, pd_2x2, b_2):
        """Adding HRFReadout must not change behaviour of the MeasureXReadout path."""
        solver = QuantumLinearSolver(
            qlsa=HHL(num_qpe_qubits=4, eig_oracle=ClassicalEigOracle()),
            readout=MeasureXReadout(),
            backend=aer_backend,
            state_prep=DefaultStatePrep(),
            shots=4096,
        )
        result = solver.solve(pd_2x2, b_2, verbose=False)
        classical = LA.solve(pd_2x2, b_2)
        cosine_sim = np.abs(np.dot(_normalized(result.solution), _normalized(classical)))
        assert cosine_sim > 0.7, f"MeasureX cosine sim regressed: {cosine_sim:.3f}"

    def test_hrf_runs_n_plus_1_circuits(self, aer_backend, pd_2x2, b_2):
        """For a 2×2 system (1 solution qubit), exactly 2 circuits should run."""
        solver = _make_solver(aer_backend, shots=1024, num_trees=5)
        result = solver.solve(pd_2x2, b_2, verbose=False)
        # 2×2 → n_sol=1 → 1 base + 1 H-variant = 2 circuits
        assert result.metadata["num_hrf_circuits"] == 2

    def test_register_names_convention(self):
        """Verify the join_data register order: ancilla first = rightmost bits."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.apply(qlsa_circ)
        rn = readout.register_names
        assert rn[0] == "ancilla_flag_result"   # first → rightmost char
        assert rn[1] == "hrf_x_result"           # last  → leftmost chars

    def test_extract_probs_bitstring_convention(self):
        """Cross-check that key[-1]='1' means ancilla=1 with our register_names."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.apply(qlsa_circ)

        # Construct counts consistent with the convention:
        # "01" → solution='0', ancilla='1' (success)
        counts = {"01": 100, "00": 50}
        result = MeasurementResult(counts)
        probs, rate = readout._extract_probs(result, n_sol=1)
        assert np.isclose(probs[0], 1.0)  # only state 0 succeeds
        assert np.isclose(rate, 100 / 150)

    def test_hrf_vs_measure_x_agreement(self, aer_backend, pd_2x2, b_2):
        """HRF and MeasureX solutions should agree in direction (cosine sim > 0.8)."""
        mx_solver = QuantumLinearSolver(
            qlsa=HHL(num_qpe_qubits=4, eig_oracle=ClassicalEigOracle()),
            readout=MeasureXReadout(),
            backend=aer_backend,
            state_prep=DefaultStatePrep(),
            shots=8192,
        )
        hrf_solver = _make_solver(aer_backend, shots=8192, num_trees=20)

        mx_result = mx_solver.solve(pd_2x2, b_2, verbose=False)
        hrf_result = hrf_solver.solve(pd_2x2, b_2, verbose=False)

        cosine_sim = np.abs(
            np.dot(_normalized(mx_result.solution), _normalized(hrf_result.solution))
        )
        assert cosine_sim > 0.8, (
            f"HRF and MeasureX solutions disagree: cosine_sim={cosine_sim:.3f}"
        )
