"""Backend-agnostic wrapper for quantum measurement results.

:class:`MeasurementResult` provides a uniform interface for extracting counts
and bitstrings regardless of whether the underlying result came from a Qiskit
``SamplerPubResult`` (IBM / Aer) or a plain ``dict[str, int]`` (Quantinuum).

Usage::

    result = executer.run(circuit, backend, shots=1024)
    counts = result.get_counts(["ancilla_flag_result", "x_result"])
    bitstrings = result.get_bitstrings(["ancilla_flag_result", "x_result"])
"""

from __future__ import annotations

from typing import Any


class MeasurementResult:
    """Uniform wrapper around a backend measurement result.

    Parameters
    ----------
    raw :
        Either a Qiskit ``SamplerPubResult`` (IBM / Aer path) or a
        ``dict[str, int]`` counts mapping (Quantinuum path).
    """

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_counts(self, register_names: list[str] | None = None) -> dict[str, int]:
        """Return a ``{bitstring: count}`` dict.

        For Qiskit results, *register_names* are joined via
        ``SamplerPubResult.join_data`` before extracting counts.  The order
        of the names determines the bit-ordering of the returned bitstrings:
        the *last* name's bits appear as the LSBs (rightmost characters).

        For dict results, *register_names* is ignored and the dict is
        returned directly.
        """
        try:
            from qiskit.primitives.containers import SamplerPubResult  # type: ignore[import]
            if isinstance(self._raw, SamplerPubResult):
                if register_names:
                    return self._raw.join_data(names=register_names).get_counts()
                return self._raw.get_counts()
        except ImportError:
            pass

        if isinstance(self._raw, dict):
            return self._raw

        raise ValueError(
            f"Unsupported result type: {type(self._raw)}.  "
            "Expected a SamplerPubResult or dict."
        )

    def get_bitstrings(self, register_names: list[str] | None = None) -> list[str]:
        """Return a flat list of bitstrings, one per shot.

        For Qiskit results, *register_names* controls the join order as in
        :meth:`get_counts`.

        For dict results, each bitstring is repeated *count* times so that
        the returned list has length equal to the total number of shots.
        """
        try:
            from qiskit.primitives.containers import SamplerPubResult  # type: ignore[import]
            if isinstance(self._raw, SamplerPubResult):
                if register_names:
                    return self._raw.join_data(names=register_names).get_bitstrings()
                return self._raw.get_bitstrings()
        except ImportError:
            pass

        if isinstance(self._raw, dict):
            result: list[str] = []
            for bs, count in self._raw.items():
                result.extend([bs] * count)
            return result

        raise ValueError(
            f"Unsupported result type: {type(self._raw)}.  "
            "Expected a SamplerPubResult or dict."
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"MeasurementResult({type(self._raw).__name__})"
