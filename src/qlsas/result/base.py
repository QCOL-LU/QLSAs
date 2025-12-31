from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Mapping



class Result(ABC):
    """
    Backend-agnostic execution result returned by an `Executer`.

    Notes:
    - Some executers return **counts** (sampling / measurement results).
    - Others may return an **expectation value** (observable evaluation).
    - `raw_result` can hold the backend-native object (e.g. a Qiskit Result).
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    raw_result: Any | None = None


class MeasurementResult(Result):
    """
    A result containing measurement counts.
    """

    counts: Mapping[str, int] = field(default_factory=dict)


class ExpectationValueResult(Result):
    """
    A result containing an expectation value (and optionally an uncertainty).
    """

    value: float = 0.0
    stddev: float | None = None


