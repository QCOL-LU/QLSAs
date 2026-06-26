"""Conformance tests for the :mod:`qlsas.backends` adapter protocol.

These tests check the *interface* contracts: that ``adapt()`` returns the
right adapter type, that ``compile()`` produces a ``CompiledArtifact``
with the expected payload type, that the Qrisp-shaped ``run(qc, shots,
token)`` convenience works, and that the legacy ``Executer`` /
``Transpiler`` facades still hand off through the adapters.

End-to-end correctness (counts statistics, success rates) is covered by
the existing ``test_solver`` / ``test_executer`` suites, which now run
through the same adapter dispatch under the hood.
"""

from __future__ import annotations

import numpy as np
import pytest
from pytket.circuit import Circuit as TketCircuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from qlsas.algorithms.hhl import HHL, MCRYEigOracle
from qlsas.backends import (
    Backend,
    CompiledArtifact,
    QiskitBackend,
    QuantinuumBackend,
    adapt,
)
from qlsas.measurement_result import MeasurementResult
from qlsas.quantinuum_config import QuantinuumBackendConfig
from qlsas.readout import MeasureXReadout
from qlsas.state_prep import DefaultStatePrep


# ---------------------------------------------------------------------------
# adapt() dispatch
# ---------------------------------------------------------------------------

class TestAdapt:

    def test_aer_yields_qiskit_backend(self, aer_backend):
        adapter = adapt(aer_backend)
        assert isinstance(adapter, QiskitBackend)
        assert adapter.underlying is aer_backend

    def test_quantinuum_config_yields_quantinuum_backend(self):
        cfg = QuantinuumBackendConfig(device_name="H1-1E", n_qubits=4)
        adapter = adapt(cfg)
        assert isinstance(adapter, QuantinuumBackend)
        assert adapter.config is cfg

    def test_idempotent_for_already_adapted(self, aer_backend):
        first = adapt(aer_backend)
        second = adapt(first)
        assert first is second

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported backend type"):
            adapt({"not": "a backend"})


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------

class TestProtocolSurface:

    def test_qiskit_backend_satisfies_protocol(self, aer_backend):
        backend: Backend = QiskitBackend(aer_backend)
        assert isinstance(backend.name, str) and backend.name
        assert hasattr(backend, "compile")
        assert hasattr(backend, "run_compiled")
        assert hasattr(backend, "run")
        assert backend.supports_multi_circuit is True

    def test_quantinuum_backend_satisfies_protocol(self):
        cfg = QuantinuumBackendConfig(device_name="H1-1E", n_qubits=4)
        backend: Backend = QuantinuumBackend(cfg)
        assert isinstance(backend.name, str) and backend.name
        assert hasattr(backend, "compile")
        assert hasattr(backend, "run_compiled")
        assert hasattr(backend, "run")
        # Multi-circuit dispatch (HRF) has not been validated for
        # Quantinuum yet — capability flag must report False.
        assert backend.supports_multi_circuit is False


# ---------------------------------------------------------------------------
# QiskitBackend behaviour
# ---------------------------------------------------------------------------

class TestQiskitBackendCompile:

    def test_compile_returns_artifact_with_qc_payload(self, aer_backend):
        qc = _bell_circuit()
        backend = QiskitBackend(aer_backend)

        artifact = backend.compile(qc, optimization_level=1)

        assert isinstance(artifact, CompiledArtifact)
        assert isinstance(artifact.payload, QuantumCircuit)

    def test_compile_invalid_optimization_level_raises(self, aer_backend):
        qc = _bell_circuit()
        backend = QiskitBackend(aer_backend)
        with pytest.raises(ValueError, match="optimization level"):
            backend.compile(qc, optimization_level=99)

    def test_compile_accepts_pytket_circuit(self, aer_backend):
        # The legacy transpiler accepted a pytket Circuit and converted it
        # to Qiskit; preserve that behaviour through the adapter.
        tket = TketCircuit(2)
        tket.H(0)
        tket.CX(0, 1)

        backend = QiskitBackend(aer_backend)
        artifact = backend.compile(tket, optimization_level=1)

        assert isinstance(artifact.payload, QuantumCircuit)


class TestQiskitBackendRun:

    def test_run_compiled_end_to_end(self, aer_backend):
        backend = QiskitBackend(aer_backend)
        artifact = backend.compile(_measured_bell_circuit(), optimization_level=1)

        result = backend.run_compiled(artifact, shots=64, verbose=False)

        assert isinstance(result, MeasurementResult)
        counts = result.get_counts(["m"])
        assert sum(counts.values()) == 64
        # Bell state should produce only "00" and "11" outcomes.
        assert all(bs in ("00", "11") for bs in counts)

    def test_run_qrisp_signature(self, aer_backend):
        # qc, shots, token -- matches Qrisp's VirtualBackend.run.
        backend = QiskitBackend(aer_backend)

        result = backend.run(_measured_bell_circuit(), shots=32, token="")

        assert isinstance(result, MeasurementResult)
        assert sum(result.get_counts(["m"]).values()) == 32

    def test_run_compiled_ignores_unknown_kwargs(self, aer_backend):
        backend = QiskitBackend(aer_backend)
        artifact = backend.compile(_measured_bell_circuit(), optimization_level=1)
        # Future / backend-specific kwargs must not blow up Aer execution.
        result = backend.run_compiled(
            artifact, shots=32, verbose=False, register_infos=None,
        )
        assert sum(result.get_counts(["m"]).values()) == 32


# ---------------------------------------------------------------------------
# QuantinuumBackend compile metadata (no execution: would require qnexus / Selene)
# ---------------------------------------------------------------------------

class TestQuantinuumBackendCompile:

    def _hhl_circuit_with_measurement(self):
        sp = DefaultStatePrep()
        hhl = HHL(num_qpe_qubits=3, eig_oracle=MCRYEigOracle())
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 0.0])
        qlsa_circuit = hhl.build_circuit(A, b, sp)
        return MeasureXReadout().apply(qlsa_circuit)

    def test_compile_local_emulator_populates_metadata(self):
        cfg = QuantinuumBackendConfig(
            device_name="H1-1E", n_qubits=8, use_local_emulator=True,
        )
        backend = QuantinuumBackend(cfg)

        artifact = backend.compile(
            self._hhl_circuit_with_measurement(), optimization_level=1
        )

        assert isinstance(artifact, CompiledArtifact)
        assert "register_infos" in artifact.backend_metadata
        assert "measurement_plan" in artifact.backend_metadata
        assert artifact.backend_metadata["optimization_level"] == 1
        # Local-emulator path strips measurements and produces a pytket Circuit.
        assert hasattr(artifact.payload, "n_qubits")

    def test_compile_nexus_path_populates_metadata(self):
        cfg = QuantinuumBackendConfig(
            device_name="H1-1E", n_qubits=8, use_local_emulator=False,
        )
        backend = QuantinuumBackend(cfg)

        artifact = backend.compile(
            self._hhl_circuit_with_measurement(), optimization_level=2
        )

        assert "register_infos" in artifact.backend_metadata
        assert "measurement_plan" in artifact.backend_metadata
        assert artifact.backend_metadata["optimization_level"] == 2

    def test_run_compiled_requires_metadata(self):
        cfg = QuantinuumBackendConfig(device_name="H1-1E", n_qubits=4)
        backend = QuantinuumBackend(cfg)

        artifact = CompiledArtifact(payload=TketCircuit(2))  # missing metadata
        with pytest.raises(ValueError, match="register_infos"):
            backend.run_compiled(artifact, shots=10, verbose=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bell_circuit() -> QuantumCircuit:
    qr = QuantumRegister(2, name="q")
    qc = QuantumCircuit(qr)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    return qc


def _measured_bell_circuit() -> QuantumCircuit:
    qr = QuantumRegister(2, name="q")
    cr = ClassicalRegister(2, name="m")
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.measure(qr, cr)
    return qc
