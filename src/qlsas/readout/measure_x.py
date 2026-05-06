"""MeasureX readout: direct computational-basis measurement of the solution register."""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit import ClassicalRegister

from qlsas.measurement_result import to_counts
from qlsas.post_processor import tomography_from_counts
from qlsas.readout.base import (
    Readout,
    QLSACircuit,
    SuccessCriterion,
    TomographyResult,
)


class MeasureXReadout(Readout):
    """Measure every qubit of the solution register in the computational basis.

    Post-processing reconstructs the solution vector via standard tomography
    (frequency counting on the successful-ancilla subspace).
    """

    # Classical registers joined for post-processing (ancilla flag first so
    # its bit is the LSB / rightmost character of each bitstring).
    _REGISTER_NAMES: list[str] = ["ancilla_flag_result", "x_result"]

    def __init__(self) -> None:
        # Stashed by apply() so process() can route post-selection through
        # the QLSA's SuccessCriterion (None for tests with synthetic counts).
        self._success_criterion: Optional[SuccessCriterion] = None

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
        self._success_criterion = qlsa_circuit.success_criterion
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
    ) -> TomographyResult:
        """Reconstruct the solution vector from measurement counts.

        Parameters
        ----------
        result : MeasurementResult or dict
            Wrapped measurement result.  A plain ``dict`` is also accepted for
            convenience (e.g. when passing already-extracted counts).

        Returns
        -------
        TomographyResult
            Iterable as ``(direction, success_rate, residual)`` for
            backward-compatible tuple unpacking.
        """
        counts = to_counts(result, self._REGISTER_NAMES)
        tr = tomography_from_counts(
            counts, A, b, success_criterion=self._success_criterion
        )

        if verbose:
            total_shots = sum(counts.values())
            num_successful = int(round(tr.success_rate * total_shots))
            print(f"total shots: {total_shots}")
            print(f"num_successful_shots: {num_successful}")
            print(f"success rate: {tr.success_rate}")
            print(f"solver residual: {tr.residual}")

        return tr
