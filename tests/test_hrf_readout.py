"""Tests for HRFReadout and the HRF solve path in QuantumLinearSolver.

Test hierarchy
--------------
Unit — pure-Python, no backend:
  TestBuildCircuits          register wiring, base + N H-variant circuits
  TestPostselectProbs        post-selection logic, edge cases
  TestRegisterNames          join_data register-order convention

Integration — uses AerSimulator, real HHL circuits:
  TestHRFSolverEndToEnd      2×2 and 4×4 systems, result quality checks
  TestHRFSolverContracts     SolveResult fields, error conditions
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as LA
import pytest

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from qlsas.readout.base import QLSACircuit, SuccessCriterion
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
    """Build a minimal mock QLSACircuit with n solution qubits.

    Mirrors HHL's register naming and includes a SuccessCriterion so the
    HRF post-selection path matches the production code path.
    """
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
            success_criterion=SuccessCriterion(
                registers=[anc_cr], required_values=["1"],
            ),
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
# Unit: HRFReadout.build_circuits()
# ===========================================================================

class TestBuildCircuits:
    """build_circuits() is the canonical multi-circuit entry point.

    Replaces the legacy apply()/build_hrf_circuits() pair; all tests of
    register wiring, circuit count, and Hadamard placement live here.
    """

    def test_returns_n_plus_1_circuits(self):
        for n in (1, 2, 3):
            readout = HRFReadout()
            qlsa_circ, _ = _make_qlsa_circuit(n=n)
            circuits = readout.build_circuits(qlsa_circ)
            assert len(circuits) == n + 1, (
                f"Expected {n + 1} circuits (1 base + {n} H-variants) for n={n}"
            )

    def test_each_is_quantum_circuit(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        for circ in readout.build_circuits(qlsa_circ):
            assert isinstance(circ, QuantumCircuit)

    def test_all_circuits_have_solution_creg(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        for circ in readout.build_circuits(qlsa_circ):
            creg_names = [cr.name for cr in circ.cregs]
            assert "hrf_x_result" in creg_names

    def test_all_circuits_preserve_ancilla_creg(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        for circ in readout.build_circuits(qlsa_circ):
            creg_names = [cr.name for cr in circ.cregs]
            assert "ancilla_flag_result" in creg_names

    def test_solution_creg_size_matches_register(self):
        for n in (1, 2, 3):
            readout = HRFReadout()
            qlsa_circ, _ = _make_qlsa_circuit(n=n)
            for circ in readout.build_circuits(qlsa_circ):
                sol_creg = next(cr for cr in circ.cregs if cr.name == "hrf_x_result")
                assert len(sol_creg) == n

    def test_base_circuit_has_no_extra_h_on_solution(self):
        """The first circuit returned is the base — no H gates on solution qubits beyond what was already in the QLSA core."""
        readout = HRFReadout()
        qlsa_circ, mock_core = _make_qlsa_circuit(n=2)
        base = readout.build_circuits(qlsa_circ)[0]

        def _h_count_on_solution(c: QuantumCircuit) -> int:
            return sum(
                1 for inst in c.data
                if inst.operation.name == "h"
                and any(
                    c.find_bit(q).registers[0][0].name == "b_to_x_register"
                    for q in inst.qubits
                )
            )

        # The mock _make_qlsa_circuit puts 2 H gates on the solution register
        # before measurement; the base circuit should have exactly that many.
        assert _h_count_on_solution(base) == _h_count_on_solution(mock_core)

    def test_h_variants_have_one_extra_h_on_correct_qubit(self):
        """Each variant circuit (after the base) has one additional H on solution_reg[i]."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        circuits = readout.build_circuits(qlsa_circ)

        def _h_targets_on_solution(c: QuantumCircuit) -> list[int]:
            targets: list[int] = []
            for inst in c.data:
                if inst.operation.name != "h":
                    continue
                for q in inst.qubits:
                    info = c.find_bit(q)
                    reg = info.registers[0][0]
                    idx = info.registers[0][1]
                    if reg.name == "b_to_x_register":
                        targets.append(idx)
            return targets

        base_targets = _h_targets_on_solution(circuits[0])
        for iq, variant in enumerate(circuits[1:]):
            v_targets = _h_targets_on_solution(variant)
            extra = sorted(v_targets)
            for t in base_targets:
                extra.remove(t)
            assert extra == [iq], (
                f"Variant {iq} should add one H on qubit {iq}; got extras {extra}"
            )

    def test_callable_multiple_times(self):
        """build_circuits() is reentrant — repeated calls give independent lists."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        c1 = readout.build_circuits(qlsa_circ)
        c2 = readout.build_circuits(qlsa_circ)
        assert len(c1) == len(c2)
        # Should be fresh copies, not the same objects.
        assert c1[0] is not c2[0]

    def test_caches_metadata_for_register_names(self):
        """build_circuits() caches the ancilla creg name; register_names returns it."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.build_circuits(qlsa_circ)
        assert readout.register_names == ["ancilla_flag_result", "hrf_x_result"]


# ===========================================================================
# Unit: HRFReadout._postselect_probs()
# ===========================================================================

class TestPostselectProbs:
    """The post-selection helper used by combine_results.

    Replaces the deleted TestExtractProbs class; same semantics, exercised
    against the canonical _postselect_probs(success_criterion=...) signature.
    """

    def _criterion(self, width: int = 1) -> SuccessCriterion:
        cr = ClassicalRegister(width, "anc")
        return SuccessCriterion(registers=[cr], required_values=["1" * width])

    def test_basic_postselection_one_qubit(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.build_circuits(qlsa_circ)  # populates _ancilla_creg_name etc.

        # Bitstring format: [sol_bit][ancilla_bit]
        # "01": sol=0, anc=1 (success) → 60 shots
        # "11": sol=1, anc=1 (success) → 40 shots
        # "00": sol=0, anc=0 (fail)    → 50 shots
        counts = {"01": 60, "11": 40, "00": 50}
        result = MeasurementResult(counts)
        probs, rate = readout._postselect_probs(result, n_sol=1, success_criterion=self._criterion())

        assert probs.shape == (2,)
        assert np.isclose(probs[0], 60 / 100)
        assert np.isclose(probs[1], 40 / 100)
        assert np.isclose(probs.sum(), 1.0)
        assert np.isclose(rate, 100 / 150)

    def test_two_qubit_solution(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=2)
        readout.build_circuits(qlsa_circ)

        counts = {
            "001": 50,   # sol=00=0, anc=1
            "011": 30,   # sol=01=1, anc=1
            "101": 15,   # sol=10=2, anc=1
            "111": 5,    # sol=11=3, anc=1
            "000": 200,  # failures
        }
        result = MeasurementResult(counts)
        probs, rate = readout._postselect_probs(result, n_sol=2, success_criterion=self._criterion())

        assert probs.shape == (4,)
        assert np.isclose(probs.sum(), 1.0)
        assert np.isclose(probs[0], 50 / 100)
        assert np.isclose(probs[1], 30 / 100)
        assert np.isclose(probs[2], 15 / 100)
        assert np.isclose(probs[3], 5 / 100)
        assert np.isclose(rate, 100 / 300)

    def test_no_successful_shots_raises(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.build_circuits(qlsa_circ)
        counts = {"00": 100, "10": 50}  # all ancilla=0
        with pytest.raises(ValueError, match="No successful ancilla shots"):
            readout._postselect_probs(MeasurementResult(counts), n_sol=1, success_criterion=self._criterion())

    def test_all_shots_successful(self):
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.build_circuits(qlsa_circ)
        counts = {"01": 75, "11": 25}
        probs, rate = readout._postselect_probs(MeasurementResult(counts), n_sol=1, success_criterion=self._criterion())
        assert np.isclose(rate, 1.0)
        assert np.isclose(probs.sum(), 1.0)

    def test_legacy_fallback_when_no_criterion(self):
        """With success_criterion=None, falls back to key[-1] == '1'."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.build_circuits(qlsa_circ)
        counts = {"01": 100, "00": 50}
        probs, rate = readout._postselect_probs(MeasurementResult(counts), n_sol=1, success_criterion=None)
        assert np.isclose(probs[0], 1.0)
        assert np.isclose(rate, 100 / 150)


# ===========================================================================
# Unit: register_names convention
# ===========================================================================

class TestRegisterNames:

    def test_default_before_build_circuits(self):
        """Before any build_circuits() call, register_names returns sensible defaults."""
        readout = HRFReadout()
        rn = readout.register_names
        assert rn[0] == "ancilla_flag_result"
        assert rn[1] == "hrf_x_result"

    def test_picks_up_qlsa_ancilla_name(self):
        """build_circuits() updates the ancilla name from the QLSACircuit."""
        readout = HRFReadout()
        qlsa_circ, _ = _make_qlsa_circuit(n=1)
        readout.build_circuits(qlsa_circ)
        rn = readout.register_names
        assert rn[0] == "ancilla_flag_result"   # first → rightmost char
        assert rn[1] == "hrf_x_result"           # last  → leftmost chars


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

    def test_hrf_vs_measure_x_agreement(self, aer_backend, pd_2x2, b_2):
        """HRF and MeasureX solutions should agree in direction (cosine sim > 0.8).

        This is the regression net for the unit-norm-direction contract:
        if HRFReadout were to silently revert to returning a pre-scaled
        vector, this test would still pass because we normalise both
        solutions before comparing — but the architecture-level
        SolveResult.direction contract is what the rest of the codebase
        relies on.
        """
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
