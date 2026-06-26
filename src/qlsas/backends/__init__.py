"""Backend adapters: a uniform interface over IBM/Aer, Quantinuum, and CUDA-Q.

The :class:`Backend` protocol shape mirrors Qrisp's
:class:`VirtualBackend.run(qc, shots, token)` so that wrapping a qlsas
backend for use with a Qrisp algorithm layer is a one-line adapter
(see :func:`as_qrisp_backend`).
"""

from typing import Any, Callable

from qlsas.backends.base import Backend, CompiledArtifact, RegisterPlan
from qlsas.backends.cudaq_backend import CudaqBackend
from qlsas.backends.qiskit_backend import QiskitBackend
from qlsas.backends.quantinuum_backend import QuantinuumBackend
from qlsas.backends.dispatch import adapt

__all__ = [
    "Backend",
    "CompiledArtifact",
    "RegisterPlan",
    "CudaqBackend",
    "QiskitBackend",
    "QuantinuumBackend",
    "adapt",
    "as_qrisp_backend",
]


def as_qrisp_backend(backend: Backend) -> Callable[..., dict[str, int]]:
    """Wrap a qlsas :class:`Backend` for use with the Qrisp algorithm layer.

    Returns a callable with the signature Qrisp's :class:`VirtualBackend`
    expects: ``run(qc, shots=None, token="") -> dict[str, int]``.

    Use::

        from qrisp.interface import VirtualBackend
        from qlsas.backends import CudaqBackend, as_qrisp_backend

        vbackend = VirtualBackend(as_qrisp_backend(CudaqBackend("nvidia-fp64")))

    The wrapper drops the richer :class:`MeasurementResult` envelope in
    favour of a plain counts dict, which is what Qrisp's
    :class:`VirtualBackend` is parameterised over.
    """

    def _qrisp_run(qc: Any, shots: int | None = None, token: str = "") -> dict[str, int]:
        result = backend.run(qc, shots=shots if shots is not None else 1024, token=token)
        return result.get_counts()

    _qrisp_run.__qualname__ = f"as_qrisp_backend({backend.name})"
    return _qrisp_run
