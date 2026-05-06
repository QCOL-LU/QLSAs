"""Architecture-level tests for the readout/post-processing refactor.

These cover the two structural pieces that did not exist before:
1. ``MultiCircuitReadout`` — the solver dispatches on the protocol, not on
   any concrete subclass. A synthetic 3-circuit readout proves this.
2. ``SuccessCriterion`` — supports multi-register success patterns, ready
   for QSVT-style solvers whose success depends on multiple ancilla bits.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qlsas.measurement_result import MeasurementResult
from qlsas.readout.base import (
    MultiCircuitReadout,
    QLSACircuit,
    SuccessCriterion,
    TomographyResult,
)


# ---------------------------------------------------------------------------
# SuccessCriterion: single + multi-register patterns
# ---------------------------------------------------------------------------

class TestSuccessCriterion:

    def test_single_register_matches_legacy(self):
        """One register of width 1, required '1' — equivalent to key[-1] == '1'."""
        cr = ClassicalRegister(1, name="anc")
        sc = SuccessCriterion(registers=[cr], required_values=["1"])
        assert sc.matches("01")
        assert sc.matches("11")
        assert not sc.matches("00")
        assert not sc.matches("10")

    def test_multi_register_two_ancillas(self):
        """QSVT readiness: two registers, e.g. widths 1 + 2, required '1' + '11'."""
        anc_a = ClassicalRegister(1, name="anc_a")
        anc_b = ClassicalRegister(2, name="anc_b")
        sc = SuccessCriterion(
            registers=[anc_a, anc_b],
            required_values=["1", "11"],
        )
        # Convention: registers[0] occupies the rightmost char, registers[1]
        # the next chars to the left. So bitstring "<rest>(b_b)(b_a)" with
        # the rightmost 1 char being anc_a and the next 2 being anc_b.
        # For success we need anc_a == '1' AND anc_b == '11'.
        # bitstring "0 11 1" (no spaces: "0111") → success
        assert sc.matches("0111")
        assert sc.matches("1111")
        # Wrong anc_a:
        assert not sc.matches("0110")
        # Wrong anc_b:
        assert not sc.matches("0011")

    def test_required_value_width_mismatch_raises(self):
        cr = ClassicalRegister(2, name="anc")
        with pytest.raises(ValueError, match="length"):
            SuccessCriterion(registers=[cr], required_values=["1"])

    def test_register_names_property(self):
        cr_a = ClassicalRegister(1, name="anc_a")
        cr_b = ClassicalRegister(1, name="anc_b")
        sc = SuccessCriterion(registers=[cr_a, cr_b], required_values=["1", "0"])
        assert sc.register_names == ["anc_a", "anc_b"]


# ---------------------------------------------------------------------------
# MeasurementResult.get_postselected_counts
# ---------------------------------------------------------------------------

class TestGetPostselectedCounts:

    def test_legacy_fallback(self):
        """When success_criterion is None, fall back to key[-1] == '1'."""
        mr = MeasurementResult({"01": 5, "11": 3, "00": 7})
        filtered, n_succ, n_total = mr.get_postselected_counts(
            ["a", "b"], success_criterion=None,
        )
        assert n_succ == 8
        assert n_total == 15
        assert filtered == {"01": 5, "11": 3}

    def test_with_criterion(self):
        cr = ClassicalRegister(1, name="anc")
        sc = SuccessCriterion(registers=[cr], required_values=["1"])
        mr = MeasurementResult({"01": 5, "11": 3, "00": 7, "10": 2})
        filtered, n_succ, n_total = mr.get_postselected_counts(["x_result", "anc"], sc)
        assert n_succ == 8
        assert n_total == 17
        assert filtered == {"01": 5, "11": 3}

    def test_multi_register_criterion(self):
        anc_a = ClassicalRegister(1, name="anc_a")
        anc_b = ClassicalRegister(1, name="anc_b")
        sc = SuccessCriterion(registers=[anc_a, anc_b], required_values=["1", "1"])
        # Bitstring layout (right-to-left in the joined bitstring):
        # rightmost 1 char = anc_a, next 1 char = anc_b, rest = readout bits
        # Success requires last 2 chars == "11".
        mr = MeasurementResult({"011": 5, "111": 3, "010": 9, "001": 1})
        filtered, n_succ, _ = mr.get_postselected_counts(
            ["anc_a", "anc_b", "x_result"], sc,
        )
        assert n_succ == 8
        assert filtered == {"011": 5, "111": 3}


# ---------------------------------------------------------------------------
# MultiCircuitReadout: synthetic 3-circuit readout proves generic dispatch
# ---------------------------------------------------------------------------

class _FakeMultiCircuit(MultiCircuitReadout):
    """Returns a fixed direction; ignores the actual measurement results.

    Exists to verify that QuantumLinearSolver dispatches on the protocol
    rather than on any concrete subclass like HRFReadout.
    """

    _SOLUTION_CREG_NAME = "fake_x"

    def __init__(self, fixed_direction: np.ndarray):
        self._direction = fixed_direction / np.linalg.norm(fixed_direction)
        self._calls_to_build = 0
        self._calls_to_combine = 0

    @property
    def register_names(self) -> list[str]:
        return ["ancilla_flag_result", self._SOLUTION_CREG_NAME]

    def build_circuits(self, qlsa_circuit: QLSACircuit) -> list[QuantumCircuit]:
        self._calls_to_build += 1
        n_sol = len(qlsa_circuit.solution_register)
        circuits = []
        for _ in range(3):
            circ = qlsa_circuit.circuit.copy()
            creg = ClassicalRegister(n_sol, name=self._SOLUTION_CREG_NAME)
            circ.add_register(creg)
            circ.measure(qlsa_circuit.solution_register, creg)
            circuits.append(circ)
        return circuits

    def combine_results(
        self, results, A, b,
        success_criterion=None, verbose=True,
    ) -> TomographyResult:
        self._calls_to_combine += 1
        from qlsas.post_processor import norm_estimation
        alpha = float(norm_estimation(A, b, self._direction))
        residual = float(np.linalg.norm(A @ (alpha * self._direction) - b))
        return TomographyResult(
            direction=self._direction,
            alpha=alpha,
            success_rate=1.0,
            residual=residual,
            metadata={"num_circuits": len(results)},
        )

    # apply()/process() are inherited from Readout; not needed for the
    # multi-circuit dispatch path. Provide stubs so the ABC instantiates.
    def apply(self, qlsa_circuit, *, state_prep=None):
        raise NotImplementedError

    def process(self, result, A, b, verbose=True):
        raise NotImplementedError


class TestMultiCircuitDispatch:

    def test_solver_dispatches_to_multi(self, aer_backend, pd_2x2, b_2, hhl_classical, state_prep):
        """The solver dispatches on the MultiCircuitReadout protocol — no
        knowledge of any specific subclass."""
        from qlsas.solver import QuantumLinearSolver
        import numpy.linalg as LA

        readout = _FakeMultiCircuit(LA.solve(pd_2x2, b_2))
        solver = QuantumLinearSolver(
            qlsa=hhl_classical,
            readout=readout,
            backend=aer_backend,
            state_prep=state_prep,
            shots=128,  # only used to wire up; results ignored by the fake
        )
        result = solver.solve(pd_2x2, b_2, verbose=False)

        assert readout._calls_to_build == 1
        assert readout._calls_to_combine == 1
        assert result.metadata.get("num_circuits") == 3
        # Direction should match what the fake returned (unit-norm)
        assert np.allclose(LA.norm(result.direction), 1.0)
        assert result.alpha is not None
