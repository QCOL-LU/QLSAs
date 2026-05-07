"""Backend adapters: a uniform interface over IBM/Aer, Quantinuum, and (later) CUDA-Q.

The :class:`Backend` protocol shape mirrors Qrisp's
:class:`VirtualBackend.run(qc, shots, token)` so that wrapping a qlsas
backend for use with a Qrisp algorithm layer is a one-line adapter.
"""

from qlsas.backends.base import Backend, CompiledArtifact, RegisterPlan
from qlsas.backends.qiskit_backend import QiskitBackend
from qlsas.backends.quantinuum_backend import QuantinuumBackend
from qlsas.backends.dispatch import adapt

__all__ = [
    "Backend",
    "CompiledArtifact",
    "RegisterPlan",
    "QiskitBackend",
    "QuantinuumBackend",
    "adapt",
]
