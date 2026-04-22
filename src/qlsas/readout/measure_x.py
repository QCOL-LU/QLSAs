"""MeasureX readout: direct computational-basis measurement of the solution register."""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister

from qlsas.readout.base import Readout, QLSACircuit
from qlsas.post_processor import Post_Processor


class MeasureXReadout(Readout):
    """Measure every qubit of the solution register in the computational basis.

    Post-processing reconstructs the solution vector via standard tomography
    (frequency counting on the successful-ancilla subspace).
    """

    # Classical registers joined for post-processing (ancilla flag last so
    # its bit is the LSB / rightmost character of each bitstring).
    _REGISTER_NAMES: list[str] = ["ancilla_flag_result", "x_result"]

    def __init__(self, post_processor: Post_Processor | None = None) -> None:
        self._pp = post_processor or Post_Processor()

    # ------------------------------------------------------------------
    # Readout interface
    # ------------------------------------------------------------------

    @property
    def register_names(self) -> list[str]:
        return self._REGISTER_NAMES

    def apply(
        self,
        qlsa_circuit: QLSACircuit,
        *,
        state_prep=None,  # not used by this readout; accepted for interface compat
    ):
        circ = qlsa_circuit.circuit.copy()
        x_result = ClassicalRegister(
            len(qlsa_circuit.solution_register), name="x_result"
        )
        circ.add_register(x_result)
        circ.measure(qlsa_circuit.solution_register, x_result)
        return circ

    def process(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """Reconstruct the solution vector from measurement counts.

        Parameters
        ----------
        result : MeasurementResult or dict
            Wrapped measurement result.  A plain ``dict`` is also accepted for
            convenience (e.g. when passing already-extracted counts).
        """
        counts = _to_counts(result, self._REGISTER_NAMES)
        solution, success_rate, residual = self._pp.tomography_from_counts(
            counts, A, b
        )

        if verbose:
            total_shots = sum(counts.values())
            num_successful = sum(v for k, v in counts.items() if k[-1] == "1")
            print(f"total shots: {total_shots}")
            print(f"num_successful_shots: {num_successful}")
            print(f"success rate: {success_rate}")
            print(f"solver residual: {residual}")

        return solution, success_rate, residual


# ---------------------------------------------------------------------------
# Module-private helper
# ---------------------------------------------------------------------------

def _to_counts(result, register_names: list[str]) -> dict[str, int]:
    """Extract a plain ``dict[str, int]`` from a *result* of any supported type."""
    if isinstance(result, dict):
        return result
    # MeasurementResult (or anything with get_counts)
    return result.get_counts(register_names)
