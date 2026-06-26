"""Qiskit / IBM / Aer backend adapter.

Wraps any Qiskit ``BackendV2`` (Aer simulator, IBM real hardware,
``IBMBackend`` from ``qiskit_ibm_runtime``) behind the unified
:class:`~qlsas.backends.base.Backend` interface.

Compile path: the Qiskit preset pass manager at the requested
``optimization_level``.

Run path: ``qiskit_ibm_runtime.SamplerV2`` — submitted into an active
``Session`` if one is supplied via ``run_compiled(..., session=...)``,
otherwise standalone job mode.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Union

from qiskit import QuantumCircuit
from qiskit.providers.backend import BackendV2
from qiskit.providers.jobstatus import JobStatus
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import IBMBackend, SamplerV2 as Sampler, Session
from pytket.circuit import Circuit
from pytket.extensions.qiskit import tk_to_qiskit

from qlsas.backends.base import Backend, CompiledArtifact
from qlsas.ibm_options import IBMExecutionOptions, apply_ibm_error_mitigation_options
from qlsas.measurement_result import MeasurementResult


class QiskitBackend(Backend):
    """Adapter around a Qiskit ``BackendV2`` / ``IBMBackend`` / ``AerSimulator``.

    Parameters
    ----------
    backend :
        The underlying Qiskit backend object to delegate execution to.
    ibm_options :
        Optional :class:`IBMExecutionOptions` applied to the sampler when
        ``backend`` is an ``IBMBackend``.  Ignored for Aer.  May be
        overridden per-call via ``run_compiled(..., ibm_options=...)``.
    """

    def __init__(
        self,
        backend: Union[BackendV2, IBMBackend, AerSimulator],
        ibm_options: Optional[IBMExecutionOptions] = None,
    ) -> None:
        self._backend = backend
        self._ibm_options = ibm_options

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        n = getattr(self._backend, "name", None)
        if callable(n):
            return n()
        if n is not None:
            return n
        return type(self._backend).__name__

    @property
    def underlying(self) -> Union[BackendV2, IBMBackend, AerSimulator]:
        """The wrapped Qiskit backend object."""
        return self._backend

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def compile(
        self,
        qc: Union[QuantumCircuit, Circuit],
        optimization_level: int = 2,
    ) -> CompiledArtifact:
        if optimization_level not in (0, 1, 2, 3):
            raise ValueError(
                f"Invalid optimization level: {optimization_level}. "
                "Must be 0, 1, 2, or 3."
            )

        if isinstance(qc, Circuit):
            qc = tk_to_qiskit(qc)
        elif not isinstance(qc, QuantumCircuit):
            raise ValueError(
                f"Invalid circuit type: {type(qc)}. "
                "Must be a qiskit QuantumCircuit or a pytket Circuit."
            )

        pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=self._backend
        )
        compiled = pm.run(qc)
        return CompiledArtifact(payload=compiled)

    def run_compiled(
        self,
        artifact: CompiledArtifact,
        shots: int = 1024,
        *,
        verbose: bool = True,
        session: Optional[Session] = None,
        ibm_options: Optional[IBMExecutionOptions] = None,
        **_unused: Any,
    ) -> MeasurementResult:
        sampler = Sampler(mode=session) if session is not None else Sampler(mode=self._backend)

        effective_options = ibm_options if ibm_options is not None else self._ibm_options
        if isinstance(self._backend, IBMBackend):
            apply_ibm_error_mitigation_options(sampler.options, effective_options)

        job = sampler.run([artifact.payload], shots=shots)
        if verbose:
            print(f">>> Job ID: {job.job_id()}")
        _wait_for_qiskit_job(job, poll_interval_s=5, verbose=verbose)
        return MeasurementResult(job.result()[0])


def _wait_for_qiskit_job(job: Any, poll_interval_s: float = 5, verbose: bool = True) -> str:
    """Poll a Qiskit job until it reaches a terminal status."""
    terminal = {"DONE", "CANCELLED", "ERROR"}

    def status_name(s: object) -> str:
        return s.name if isinstance(s, JobStatus) else str(s)

    status = job.status()
    last_len = 0
    while status_name(status) not in terminal:
        if verbose:
            msg = f">>> Job Status: {status_name(status)}..."
            pad = " " * max(0, last_len - len(msg))
            print(f"\r{msg}{pad}", end="", flush=True)
            last_len = len(msg)
        time.sleep(poll_interval_s)
        status = job.status()

    final = status_name(status)
    if verbose:
        msg = f">>> Job Status: {final}"
        pad = " " * max(0, last_len - len(msg))
        print(f"\r{msg}{pad}", flush=True)
    return final
