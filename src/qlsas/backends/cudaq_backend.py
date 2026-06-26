"""CUDA-Q GPU-accelerated execution backend.

CUDA-Q is officially supported on Linux (``x86_64`` / ``aarch64``) with a
CUDA-capable GPU.  The ``qpp-cpu`` target works on CPU and is the
parity-test baseline for machines without a GPU.

Install with::

    pip install qlsas[cudaq]

The adapter translates Qiskit circuits to CUDA-Q kernels at the
*execution* boundary — algorithm code (HHL, QPE, eigenvalue oracles)
stays Qiskit-native, and CUDA-Q is a pure execution layer.

Translation strategy
--------------------
1. Qiskit ``transpile`` decomposes high-level gates (``HamiltonianGate``,
   ``StatePreparation``, controlled boxes) into a basis CUDA-Q ingests.
2. ``cudaq.from_qiskit_circuit`` (preferred) converts the Qiskit circuit
   to a CUDA-Q kernel.  Falls back to a QASM 2 round-trip via
   ``cudaq.make_kernel_from_qasm`` / ``cudaq.synthesize`` if the direct
   conversion is not available on the installed CUDA-Q version.

Counts assembly
---------------
CUDA-Q's :class:`SampleResult` is a dict-like ``{joined_bitstring:
count}`` in which the **first**-measured register sits at the *leftmost*
(MSB) of the joined string; Qiskit's :meth:`SamplerPubResult.join_data`
places the first-named register at the *rightmost* (LSB).  A whole-
string reversal (``bitstring[::-1]``) flips between the two conventions
and — as a happy algebraic side effect — also flips the within-register
bit order, which the two frameworks disagree on as well.  So a single
reversal is sufficient.  This is corroborated by the parallel
``cuda-q-refactor`` branch's post-processor, whose explicit ``key[0]``
ancilla check + ``key[-1:0:-1]`` solution-register reverse is the same
transformation, only embedded in the readout.  See
``CudaqBackend._counts_from_sample_result`` for details.

The class-level ``REVERSE_BITSTRINGS`` flag (default ``True``) controls
this; the Bell-state byte-exact parity test in
``tests/test_cudaq_backend.py`` is the regression net if a future
CUDA-Q upgrade ever flips this convention.

Process-global state
--------------------
``cudaq.set_target`` is process-global.  A module-level
:class:`threading.Lock` serialises the ``set_target`` → ``sample``
sequence so that two :class:`CudaqBackend` instances with different
targets cannot race.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from qiskit import QuantumCircuit, qasm2, transpile

from qlsas.backends.base import Backend, CompiledArtifact, RegisterPlan
from qlsas.measurement_result import MeasurementResult


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_CUDAQ_TARGET_LOCK = threading.Lock()
"""Serialises cudaq.set_target across CudaqBackend instances within a process."""

_QISKIT_BASIS_GATES: tuple[str, ...] = (
    "cx", "u3", "u2", "u1", "rz", "rx", "ry", "h",
    "x", "y", "z", "s", "sdg", "t", "tdg", "measure",
)

_SUPPORTED_TARGETS: frozenset[str] = frozenset({
    "nvidia",         # single-GPU statevector, fp32 default
    "nvidia-fp64",    # single-GPU statevector, fp64
    "nvidia-mgpu",    # multi-node statevector for >34-qubit circuits
    "nvidia-mqpu",    # multi-QPU parallelism (pairs with future run_batch override)
    "qpp-cpu",        # CPU baseline; required for parity tests on GPU-less CI
})


def _import_cudaq() -> Any:
    """Lazy ``import cudaq`` with a friendly error message."""
    try:
        import cudaq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "CUDA-Q is required for CudaqBackend. Install with:\n"
            "    pip install qlsas[cudaq]\n"
            "Note: cuda-quantum is officially supported on Linux only "
            "(x86_64 / aarch64) with a CUDA-capable GPU. The 'qpp-cpu' "
            "target works on CPU."
        ) from exc
    return cudaq


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class CudaqBackend(Backend):
    """GPU-accelerated quantum circuit simulation via NVIDIA's CUDA-Q.

    Parameters
    ----------
    target :
        CUDA-Q execution target.  Must be one of: ``"nvidia"``,
        ``"nvidia-fp64"``, ``"nvidia-mgpu"``, ``"nvidia-mqpu"``,
        ``"qpp-cpu"``.
    seed :
        Optional random seed forwarded to ``cudaq.set_random_seed``;
        when set, the same seed is reapplied on every ``run_compiled``.

    Examples
    --------
    >>> from qlsas.backends import CudaqBackend
    >>> backend = CudaqBackend("nvidia-fp64")          # GPU, double precision
    >>> backend = CudaqBackend("qpp-cpu", seed=42)     # CPU, deterministic
    """

    #: Whether to reverse each CUDA-Q bitstring to match Qiskit's join_data
    #: convention (first creg → rightmost / LSB).  See the
    #: ``test_bell_state_byte_exact`` parity test.
    REVERSE_BITSTRINGS: bool = True

    def __init__(self, target: str = "nvidia", *, seed: Optional[int] = None):
        if target not in _SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported CUDA-Q target {target!r}. Pick one of "
                f"{sorted(_SUPPORTED_TARGETS)}."
            )
        self._target = target
        self._seed = seed

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"cudaq:{self._target}"

    @property
    def target(self) -> str:
        return self._target

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_multi_circuit(self) -> bool:
        # The standard per-artifact run_compiled loop in
        # QuantumLinearSolver._solve_multi works for CUDA-Q out of the
        # box; an async MQPU run_batch override is a future optimisation,
        # not a correctness requirement.
        return True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def compile(
        self,
        qc: QuantumCircuit,
        optimization_level: int = 2,
    ) -> CompiledArtifact:
        cudaq = _import_cudaq()

        # 1. Decompose to a CUDA-Q-ingestible basis. The recursive
        #    decompose flattens HamiltonianGate, StatePreparation, and
        #    controlled boxes that QASM 2 cannot represent.
        decomposed = transpile(
            qc,
            basis_gates=list(_QISKIT_BASIS_GATES),
            optimization_level=optimization_level,
        )
        flat = decomposed.decompose(reps=4)

        # 2. Qiskit → CUDA-Q kernel.
        kernel = self._qiskit_to_cudaq_kernel(cudaq, flat)

        register_names = [creg.name for creg in flat.cregs]
        register_sizes = {creg.name: len(creg) for creg in flat.cregs}
        total_bits = sum(register_sizes.values())

        return CompiledArtifact(
            payload=kernel,
            register_plan=RegisterPlan(register_names=register_names),
            backend_metadata={
                "register_names": register_names,
                "register_sizes": register_sizes,
                "total_bits": total_bits,
            },
        )

    @staticmethod
    def _qiskit_to_cudaq_kernel(cudaq: Any, qc: QuantumCircuit) -> Any:
        """Try every documented Qiskit → CUDA-Q conversion path.

        The exact attribute name has shifted across CUDA-Q minor
        releases.  We probe ``from_qiskit_circuit`` (preferred), then
        ``qiskit_to_cudaq``, then fall back to a QASM 2 round-trip via
        ``make_kernel_from_qasm`` / ``synthesize``.
        """
        errors: list[str] = []

        for attr in ("from_qiskit_circuit", "qiskit_to_cudaq"):
            convert = getattr(cudaq, attr, None)
            if convert is None:
                continue
            try:
                return convert(qc)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"cudaq.{attr}: {type(exc).__name__}: {exc}")

        # QASM 2 round-trip fallback.
        try:
            qasm_str = qasm2.dumps(qc)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"qasm2.dumps: {type(exc).__name__}: {exc}")
            qasm_str = None

        if qasm_str is not None:
            for attr in ("make_kernel_from_qasm", "synthesize"):
                convert = getattr(cudaq, attr, None)
                if convert is None:
                    continue
                try:
                    return convert(qasm_str)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"cudaq.{attr}: {type(exc).__name__}: {exc}")

        joined = "\n  ".join(errors) if errors else "(no conversion paths found on this cudaq install)"
        raise RuntimeError(
            "Every Qiskit→CUDA-Q conversion path failed:\n  "
            + joined
            + "\nUpdate CudaqBackend._qiskit_to_cudaq_kernel for this CUDA-Q version."
        )

    def run_compiled(
        self,
        artifact: CompiledArtifact,
        shots: int = 1024,
        *,
        verbose: bool = True,
        **_unused: Any,
    ) -> MeasurementResult:
        cudaq = _import_cudaq()

        with _CUDAQ_TARGET_LOCK:
            # Fail fast if the target isn't built into this CUDA-Q install
            # (e.g. asking for "nvidia" on a GPU-less box).  Skip when the
            # current target already matches — set_target on the same
            # target is wasteful and noisy.
            if cudaq.get_target().name != self._target:
                if not cudaq.has_target(self._target):
                    raise RuntimeError(
                        f"CUDA-Q target {self._target!r} is not available in this "
                        f"cuda-quantum install. Pick one supported by your build "
                        f"(qpp-cpu always works on CPU; nvidia* targets need a "
                        f"CUDA-capable GPU)."
                    )
                cudaq.set_target(self._target)
            if self._seed is not None:
                cudaq.set_random_seed(self._seed)
            if verbose:
                print(f">>> Running on {self.name} (shots={shots})")
            sample_result = cudaq.sample(artifact.payload, shots_count=shots)

        counts = self._counts_from_sample_result(sample_result)
        return MeasurementResult(counts)

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    @classmethod
    def _counts_from_sample_result(cls, sample_result: Any) -> dict[str, int]:
        """Translate CUDA-Q's :class:`SampleResult` to a counts dict.

        CUDA-Q places the *first*-measured register at the **leftmost
        (MSB)** of the joined bitstring; Qiskit's
        :meth:`SamplerPubResult.join_data` places the first-named register
        at the **rightmost (LSB)**.  Reversing the entire string flips
        between the two conventions and — by happy algebra — also
        flips the within-register bit order, which Qiskit/CUDA-Q also
        disagree on.  So a single ``bitstring[::-1]`` is sufficient.

        This convention is corroborated by the parallel cuda-q-refactor
        branch's post-processor, which checks ``key[0] == '1'`` for the
        ancilla and ``int(key[-1:0:-1], base=2)`` for the solution
        register — exactly the layout that ``[::-1]`` translates from.

        Bell-state byte-exact parity test
        (``test_cudaq_backend.py::TestBellParity``) is the regression net
        if a future CUDA-Q upgrade ever flips this; the fix is to set
        ``REVERSE_BITSTRINGS = False``.
        """
        try:
            pairs = list(sample_result.items())
        except AttributeError:
            # Older CUDA-Q SampleResult: dict-like via iteration + indexing.
            pairs = [(bs, sample_result[bs]) for bs in sample_result]

        counts: dict[str, int] = {}
        for bitstring, count in pairs:
            key = bitstring[::-1] if cls.REVERSE_BITSTRINGS else bitstring
            counts[key] = int(count)
        return counts
