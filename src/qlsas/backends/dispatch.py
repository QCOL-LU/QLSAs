"""Adapter dispatch: turn a raw backend object into a :class:`Backend`.

Existing callers (``QuantumLinearSolver``, examples, notebooks) hand the
solver a *raw* backend object — an Aer simulator, an ``IBMBackend``, or a
``QuantinuumBackendConfig``.  :func:`adapt` wraps that raw object in the
appropriate :class:`~qlsas.backends.base.Backend` adapter so the rest of
the pipeline can talk to a single interface.

Pre-wrapped :class:`Backend` instances are returned unchanged, so the
function is idempotent.
"""

from __future__ import annotations

from typing import Optional

from qiskit.providers.backend import BackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend

from qlsas.backends.base import Backend
from qlsas.backends.qiskit_backend import QiskitBackend
from qlsas.backends.quantinuum_backend import QuantinuumBackend
from qlsas.ibm_options import IBMExecutionOptions
from qlsas.quantinuum_config import QuantinuumBackendConfig


def adapt(
    backend: object,
    *,
    ibm_options: Optional[IBMExecutionOptions] = None,
) -> Backend:
    """Return a :class:`Backend` adapter for *backend*.

    *ibm_options* is forwarded to :class:`QiskitBackend` and ignored for
    other adapter types.
    """
    if isinstance(backend, Backend):
        return backend
    if isinstance(backend, (BackendV2, IBMBackend, AerSimulator)):
        return QiskitBackend(backend, ibm_options=ibm_options)
    if isinstance(backend, QuantinuumBackendConfig):
        return QuantinuumBackend(backend)
    raise ValueError(f"Unsupported backend type: {type(backend).__name__}")
