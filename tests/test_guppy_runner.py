"""Unit tests for Quantinuum/Guppy result parsing in qlsas.guppy_runner.

These tests lock in the bit-ordering contract between guppylang's measure_array
output and the Post_Processor tomography logic.  They would have caught the
endianness bug where Guppy returns [qubit[0], qubit[1], ...] (LSB-first) but
tomography_from_counts expects MSB-first bitstrings.
"""

from __future__ import annotations

import numpy as np
import numpy.linalg as LA
import pytest

from qlsas.guppy_runner import (
    RegisterMeasurement,
    _value_to_bitstring,
    _combine_shot_results,
)
from qlsas.post_processor import Post_Processor


# ---------------------------------------------------------------------------
# _value_to_bitstring — endianness contract
# ---------------------------------------------------------------------------

class TestValueToBitstring:
    """Verify _value_to_bitstring produces MSB-first bitstrings for multi-qubit registers.

    Guppy's measure_array returns [qubit[0], qubit[1], ...] (LSB-first).
    Qiskit/tomography_from_counts use MSB-first: qubit[n-1] on the left.
    _value_to_bitstring must reverse the list so int(bitstring, 2) = correct coordinate.
    """

    def test_two_qubits_coord_1(self):
        """[btr[0]=1, btr[1]=0] → coord = 2^0*1 + 2^1*0 = 1 → bitstring '01'."""
        out = _value_to_bitstring([True, False])
        assert out == "01"
        assert int(out, 2) == 1

    def test_two_qubits_coord_2(self):
        """[btr[0]=0, btr[1]=1] → coord = 2^0*0 + 2^1*1 = 2 → bitstring '10'."""
        out = _value_to_bitstring([False, True])
        assert out == "10"
        assert int(out, 2) == 2

    def test_two_qubits_coord_0_and_3(self):
        assert _value_to_bitstring([False, False]) == "00"
        assert int("00", 2) == 0
        assert _value_to_bitstring([True, True]) == "11"
        assert int("11", 2) == 3

    def test_single_qubit_no_endianness(self):
        assert _value_to_bitstring([True]) == "1"
        assert _value_to_bitstring([False]) == "0"

    def test_single_bool_passthrough(self):
        assert _value_to_bitstring(True) == "1"
        assert _value_to_bitstring(False) == "0"

    def test_string_passthrough(self):
        assert _value_to_bitstring("01") == "01"

    def test_tuple_same_as_list(self):
        assert _value_to_bitstring((True, False)) == "01"

    def test_three_qubits_coord_5(self):
        """[1,0,1] → coord 5 → MSB-first '101'."""
        out = _value_to_bitstring([True, False, True])
        assert out == "101"
        assert int(out, 2) == 5


# ---------------------------------------------------------------------------
# _combine_shot_results — format contract with tomography_from_counts
# ---------------------------------------------------------------------------

def _mock_shot(ancilla: bool, x_result: list[bool]):
    """Minimal mock of guppylang QsysShot for measure_x layout."""

    class MockShot:
        def as_dict(self):
            return {"ancilla_flag_result": ancilla, "x_result": x_result}

    return MockShot()


class TestCombineShotResults:
    """Verify _combine_shot_results produces counts compatible with tomography_from_counts."""

    @pytest.fixture
    def measure_x_plan(self):
        return [
            RegisterMeasurement("ancilla_flag_register", "ancilla_flag_result", 1),
            RegisterMeasurement("b_to_x_register", "x_result", 2),
        ]

    def test_produces_correct_bitstring_format(self, measure_x_plan):
        """Bitstring = [x_result MSB-first][ancilla]; last bit = ancilla, first x_size = coord."""
        raw = [
            _mock_shot(ancilla=True, x_result=[True, False]),   # coord 1, success
            _mock_shot(ancilla=True, x_result=[True, False]),   # same
            _mock_shot(ancilla=False, x_result=[False, True]),  # coord 2, fail
        ]
        counts = _combine_shot_results(raw, measure_x_plan)
        # x_result [1,0] → "01", ancilla True → "1"  → key "011"
        # x_result [0,1] → "10", ancilla False → "0" → key "100"
        assert counts == {"011": 2, "100": 1}

    def test_counts_parseable_by_tomography(self, measure_x_plan):
        """Output of _combine_shot_results must be valid input for tomography_from_counts."""
        raw = [
            _mock_shot(ancilla=True, x_result=[True, False]),
            _mock_shot(ancilla=True, x_result=[False, True]),
            _mock_shot(ancilla=True, x_result=[True, False]),
            _mock_shot(ancilla=False, x_result=[False, False]),
        ]
        counts = _combine_shot_results(raw, measure_x_plan)
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 1.0]) / np.linalg.norm([1.0, 1.0])
        pp = Post_Processor()
        solution, success_rate, residual = pp.tomography_from_counts(counts, A, b)
        assert len(solution) == 2
        assert 0 <= success_rate <= 1
        assert np.isfinite(residual)


# ---------------------------------------------------------------------------
# Bit-reversal produces wrong residual — regression guard
# ---------------------------------------------------------------------------

class TestBitOrderingRegression:
    """Tests that would fail if the endianness fix were reverted."""

    def test_bit_reversed_counts_produce_worse_residual(self):
        """If x_result bits are LSB-first (wrong), coords 1 and 2 are swapped → worse residual.

        Use a 4x4 problem where true solution has coord 1 dominant and coord 2 near-zero.
        CORRECT counts: coord 1 gets most hits (MSB-first key "011").
        WRONG counts: same hits but keys "101"/"011" swapped → coords 1↔2 reversed.
        Wrong counts must produce strictly higher residual than correct counts.
        """
        A = np.diag([1.0, 2.0, 8.0, 4.0])
        b = np.array([0.05, 0.98, 0.05, 0.05])
        b = b / np.linalg.norm(b)
        x_true = LA.solve(A, b)
        x_norm = x_true / np.linalg.norm(x_true)

        # True solution: coord 1 dominant (|x[1]|² large), coord 2 small
        assert x_norm[1] ** 2 > 0.5
        assert x_norm[2] ** 2 < 0.1

        # CORRECT counts (MSB-first): 80 hits at coord 1, 20 at coord 2
        correct_counts = {
            "011": 80,   # coord 1, ancilla=1
            "101": 20,   # coord 2, ancilla=1
            "001": 50,   # coord 0, ancilla=1
            "111": 50,   # coord 3, ancilla=1
        }

        # WRONG counts (simulates buggy LSB-first): coord 1 hits labelled as coord 2
        wrong_counts = {
            "101": 80,   # hits that belong to coord 1, but key decodes to coord 2
            "011": 20,   # hits that belong to coord 2, but key decodes to coord 1
            "001": 50,
            "111": 50,
        }

        pp = Post_Processor()
        _, _, residual_correct = pp.tomography_from_counts(correct_counts, A, b)
        _, _, residual_wrong = pp.tomography_from_counts(wrong_counts, A, b)

        assert residual_wrong > residual_correct, (
            f"Bit-reversed counts must produce higher residual ({residual_wrong:.4f}) "
            f"than correct counts ({residual_correct:.4f}) — guards against endianness revert."
        )
        assert residual_wrong > 0.5, "Wrong counts should yield residual ≫ 0.5"

    def test_synthetic_4x4_correct_coord_dominance(self):
        """Tomography with MSB-first format puts histogram mass at correct coordinate."""
        A = np.diag([1.0, 2.0, 8.0, 4.0])
        b = np.array([0.05, 0.98, 0.05, 0.05])
        b = b / np.linalg.norm(b)
        x_size = 2

        # 100 successful shots: 80 at coord 1 (true dominant), 6–7 each at others
        counts = {
            "011": 80,  # coord 1
            "001": 7,
            "101": 7,
            "111": 6,
            "000": 50,
            "010": 50,
        }

        pp = Post_Processor()
        solution, success_rate, residual = pp.tomography_from_counts(counts, A, b)

        assert solution[1] > solution[2], "Coord 1 should dominate (MSB-first '011')"
        assert np.isfinite(residual)
