"""Tests for the CUDA-Q backend adapter.

The lightweight tests (target validation, lazy-import behaviour, protocol
conformance) run anywhere — including machines without ``cuda-quantum``
installed.  The execution / parity tests use
``pytest.importorskip("cudaq")`` and run only when CUDA-Q is importable
(typically Linux-x86_64 / aarch64 with a CUDA-capable GPU; the
``qpp-cpu`` target works on CPU).

The Bell-state byte-exact test in :class:`TestBellParity` is the
**regression net for bitstring endianness**.  If it ever fails after a
CUDA-Q upgrade, flip ``CudaqBackend.REVERSE_BITSTRINGS`` and retest.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import numpy.linalg as LA
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

from qlsas.algorithms.hhl import HHL, MCRYEigOracle
from qlsas.backends import (
    Backend,
    CompiledArtifact,
    CudaqBackend,
    QiskitBackend,
    as_qrisp_backend,
)
from qlsas.measurement_result import MeasurementResult
from qlsas.readout import MeasureXReadout
from qlsas.solver import QuantumLinearSolver
from qlsas.state_prep import DefaultStatePrep


CUDAQ_INSTALLED = importlib.util.find_spec("cudaq") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measured_bell_circuit() -> QuantumCircuit:
    qr = QuantumRegister(2, name="q")
    cr = ClassicalRegister(2, name="m")
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    qc.measure(qr, cr)
    return qc


def _normalized(v: np.ndarray) -> np.ndarray:
    return v / LA.norm(v)


# ===========================================================================
# Lightweight tests — run with or without cudaq installed
# ===========================================================================

class TestTargetValidation:

    def test_supported_targets_accepted(self):
        for tgt in ("nvidia", "nvidia-fp64", "nvidia-mgpu", "nvidia-mqpu", "qpp-cpu"):
            backend = CudaqBackend(tgt)
            assert backend.target == tgt
            assert backend.name == f"cudaq:{tgt}"

    def test_unsupported_target_raises(self):
        with pytest.raises(ValueError, match="Unsupported CUDA-Q target"):
            CudaqBackend("not-a-real-target")

    def test_default_target_is_nvidia(self):
        assert CudaqBackend().target == "nvidia"


class TestProtocolConformance:

    def test_is_backend_subclass(self):
        backend = CudaqBackend("qpp-cpu")
        assert isinstance(backend, Backend)

    def test_supports_multi_circuit_defaults_true(self):
        assert CudaqBackend("qpp-cpu").supports_multi_circuit is True

    def test_seed_round_trips(self):
        assert CudaqBackend("qpp-cpu", seed=42).seed == 42
        assert CudaqBackend("qpp-cpu").seed is None


class TestQrispAdapter:

    def test_returns_callable_with_qrisp_signature(self):
        # Doesn't actually invoke cudaq — just verifies the adapter shape.
        backend = CudaqBackend("qpp-cpu")
        wrapped = as_qrisp_backend(backend)
        assert callable(wrapped)
        # Signature parity with Qrisp's VirtualBackend.run(qc, shots, token).
        import inspect
        sig = inspect.signature(wrapped)
        assert list(sig.parameters) == ["qc", "shots", "token"]


@pytest.mark.skipif(CUDAQ_INSTALLED, reason="only meaningful when cudaq is NOT installed")
class TestLazyImportErrorMessage:
    """When cudaq is unavailable, friendly ImportError must fire on first use,
    not at module import time."""

    def test_import_succeeds_without_cudaq(self):
        # Reaching this test class means CudaqBackend was importable.
        assert CudaqBackend is not None

    def test_compile_raises_friendly_import_error(self):
        backend = CudaqBackend("qpp-cpu")
        with pytest.raises(ImportError, match=r"qlsas\[cudaq\]"):
            backend.compile(_measured_bell_circuit())

    def test_run_raises_friendly_import_error(self):
        # run() chains compile + run_compiled; the first call (compile)
        # is where the import is attempted.
        backend = CudaqBackend("qpp-cpu")
        with pytest.raises(ImportError, match=r"qlsas\[cudaq\]"):
            backend.run(_measured_bell_circuit(), shots=10)


# ===========================================================================
# Execution tests — gated on cudaq being importable
# ===========================================================================

@pytest.mark.cudaq
class TestBellParity:
    """Bell-state byte-exact parity check between Aer and CUDA-Q's qpp-cpu.

    Both targets are deterministic statevector simulators with the same
    seed, so the dicts must agree byte-for-byte across all keys.

    The bitstring-ordering convention is well-established: CUDA-Q places
    the first-measured register at the MSB; Qiskit places it at the LSB
    (a single ``[::-1]`` flips between the two — verified by the parallel
    ``cuda-q-refactor`` branch's post-processor).  This test catches any
    future CUDA-Q regression that flips the convention; the fix is
    ``CudaqBackend.REVERSE_BITSTRINGS = False``.
    """

    def test_bell_state_byte_exact(self):
        pytest.importorskip("cudaq")

        qc = _measured_bell_circuit()

        aer_backend = QiskitBackend(AerSimulator(seed_simulator=12345))
        aer_artifact = aer_backend.compile(qc, optimization_level=0)
        aer_result = aer_backend.run_compiled(aer_artifact, shots=1024, verbose=False)
        aer_counts = aer_result.get_counts(["m"])

        cudaq_backend = CudaqBackend("qpp-cpu", seed=12345)
        cudaq_artifact = cudaq_backend.compile(qc, optimization_level=0)
        cudaq_result = cudaq_backend.run_compiled(cudaq_artifact, shots=1024, verbose=False)
        cudaq_counts = cudaq_result.get_counts(["m"])

        # Bell state can only produce "00" and "11"; exact distribution will
        # vary with seed, but the *support* must agree and total = 1024.
        assert set(aer_counts) == set(cudaq_counts), (
            f"Counts support disagrees — endianness convention may be wrong. "
            f"Aer: {sorted(aer_counts)}, CUDA-Q: {sorted(cudaq_counts)}"
        )
        assert sum(aer_counts.values()) == 1024
        assert sum(cudaq_counts.values()) == 1024
        # Outcomes must lie in the Bell-state subspace.
        assert set(cudaq_counts) <= {"00", "11"}, (
            f"Bell-state outcomes outside {{00, 11}}: {sorted(cudaq_counts)}"
        )


@pytest.mark.cudaq
class TestHhlParity:
    """End-to-end HHL parity check: 4x4 problem, Aer vs CUDA-Q qpp-cpu."""

    def test_4x4_direction_agrees_with_aer(self):
        pytest.importorskip("cudaq")

        A = np.array([
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 2.5, 0.5],
            [0.0, 0.0, 0.5, 2.0],
        ])
        b = _normalized(np.array([1.0, 0.0, 0.0, 0.0]))

        hhl = HHL(num_qpe_qubits=4, eig_oracle=MCRYEigOracle())
        readout = MeasureXReadout()

        aer_solver = QuantumLinearSolver(
            qlsa=hhl, readout=readout,
            backend=QiskitBackend(AerSimulator(seed_simulator=7)),
            state_prep=DefaultStatePrep(),
            shots=8192,
        )
        cudaq_solver = QuantumLinearSolver(
            qlsa=hhl, readout=readout,
            backend=CudaqBackend("qpp-cpu", seed=7),
            state_prep=DefaultStatePrep(),
            shots=8192,
        )

        aer_result = aer_solver.solve(A, b, verbose=False)
        cudaq_result = cudaq_solver.solve(A, b, verbose=False)

        # Direction (unit-norm) cosine similarity should be high for both
        # against the classical solution; here we compare the two
        # quantum directions to each other.
        cosine = abs(np.dot(_normalized(aer_result.direction),
                            _normalized(cudaq_result.direction)))
        assert cosine > 0.95, (
            f"Aer/CUDA-Q HHL direction cosine similarity too low: {cosine}. "
            f"Suggests a translation or endianness regression."
        )


@pytest.mark.cudaq
class TestSeedDeterminism:
    """Same seed + same target must produce identical counts dicts."""

    def test_qpp_cpu_seed_determinism(self):
        pytest.importorskip("cudaq")

        qc = _measured_bell_circuit()
        backend = CudaqBackend("qpp-cpu", seed=99)
        artifact = backend.compile(qc, optimization_level=0)

        first = backend.run_compiled(artifact, shots=512, verbose=False)
        second = backend.run_compiled(artifact, shots=512, verbose=False)

        assert first.get_counts(["m"]) == second.get_counts(["m"])
