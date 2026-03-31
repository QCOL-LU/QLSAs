"""Quantinuum backend bridge: local Selene emulation and Nexus cloud execution.

Two execution paths are supported:

Local Selene (``use_local_emulator=True``)
    Qiskit circuit → pytket (measurement-free, optimised) → Guppy wrapper →
    Selene emulator.  Optimisation is applied locally using pytket passes.

Nexus cloud (``use_local_emulator=False``)
    Qiskit circuit → pytket (with measurements, minimal preprocessing) →
    ``qnx.start_compile_job`` (device-native compilation on Nexus) →
    ``qnx.start_execute_job`` → ``BackendResult`` → counts dict.

Both paths return a ``dict[str, int]`` counts dict compatible with
:class:`~qlsas.post_processor.Post_Processor`.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from pytket.circuit import Circuit, OpType
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.passes import (
    DecomposeBoxes,
    AutoRebase,
    RemoveRedundancies,
    FullPeepholeOptimise,
    SquashTK1,
)

from qlsas.quantinuum_config import QuantinuumBackendConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegisterMeasurement:
    """Maps a quantum register to the classical register it is measured into."""
    quantum_register_name: str
    classical_register_name: str
    size: int


@dataclass
class RegisterInfo:
    """Name and size of a quantum register (lexicographic order)."""
    name: str
    size: int


# ---------------------------------------------------------------------------
# Measurement plan extraction (run on the ORIGINAL Qiskit circuit)
# ---------------------------------------------------------------------------

def extract_measurement_plan(circuit: QuantumCircuit) -> list[RegisterMeasurement]:
    """Determine which quantum registers are measured and into which classical registers."""
    creg_to_qreg: dict[str, str] = {}
    creg_sizes: dict[str, int] = {}

    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        qreg_name = _register_name_for_bit(circuit.qregs, instruction.qubits[0])
        creg_name = _register_name_for_bit(circuit.cregs, instruction.clbits[0])

        if creg_name not in creg_to_qreg:
            creg_to_qreg[creg_name] = qreg_name
        creg_sizes[creg_name] = creg_sizes.get(creg_name, 0) + 1

    return [
        RegisterMeasurement(
            quantum_register_name=creg_to_qreg[creg],
            classical_register_name=creg,
            size=creg_sizes[creg],
        )
        for creg in creg_to_qreg
    ]


def _register_name_for_bit(registers, bit) -> str:
    for reg in registers:
        if bit in reg:
            return reg.name
    raise ValueError(f"Bit {bit} not found in any register.")


# ---------------------------------------------------------------------------
# Circuit preparation: Selene path (measurement-free, locally optimised)
# ---------------------------------------------------------------------------

# CX, Rz, H are universally supported by guppylang / Selene and produce
# correct simulation results.  ZZPhase / PhasedX are the native Quantinuum
# hardware gates but are not needed here — for cloud execution the Nexus
# compile job handles gate-set conversion independently.
_QUANTINUUM_GATESET = {OpType.CX, OpType.Rz, OpType.H}


def prepare_pytket_circuit(
    circuit: QuantumCircuit,
    optimization_level: int = 2,
) -> tuple[Circuit, list[RegisterInfo]]:
    """Convert a Qiskit circuit to a measurement-free pytket circuit for Selene.

    Steps:
      1. Pre-transpile with ``AerSimulator`` to decompose high-level gates.
      2. Strip measurements and barriers (Guppy adds its own measurement wrapper).
      3. Convert to pytket.
      4. ``DecomposeBoxes`` to flatten any remaining box structures.
      5. ``RemoveRedundancies`` at level ≥ 1 (pre-rebase).
      6. ``FullPeepholeOptimise`` at level ≥ 2 (CX target, allow_swaps=False).
      7. ``AutoRebase`` to the Quantinuum-compatible gate set (CX, Rz, H).
      8. ``RemoveRedundancies`` at level ≥ 2 (post-rebase cleanup).
    """
    decomposed = transpile(circuit, AerSimulator(), optimization_level=1)
    stripped = _strip_measurements(decomposed)
    pytket_circuit = qiskit_to_tk(stripped)

    DecomposeBoxes().apply(pytket_circuit)
    _apply_optimization_passes(pytket_circuit, optimization_level)
    AutoRebase(_QUANTINUUM_GATESET).apply(pytket_circuit)
    if optimization_level >= 2:
        RemoveRedundancies().apply(pytket_circuit)

    return pytket_circuit, _extract_register_infos(stripped)


# ---------------------------------------------------------------------------
# Circuit preparation: Nexus path (measurements kept, minimal preprocessing)
# ---------------------------------------------------------------------------

def prepare_pytket_circuit_for_nexus(
    circuit: QuantumCircuit,
) -> tuple[Circuit, list[RegisterInfo]]:
    """Convert a Qiskit circuit to a pytket circuit for Nexus cloud execution.

    Keeps measurements intact and applies only the minimal preprocessing
    needed before upload (Aer decomposition of high-level gates, pytket
    conversion, box decomposition).  Circuit optimisation is delegated
    entirely to a Nexus compile job, which applies the device-native
    compilation pass.
    """
    decomposed = transpile(circuit, AerSimulator(), optimization_level=1)
    pytket_circuit = qiskit_to_tk(decomposed)   # measurements preserved
    DecomposeBoxes().apply(pytket_circuit)

    stripped = _strip_measurements(decomposed)
    return pytket_circuit, _extract_register_infos(stripped)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy with Measure ops, barriers, and classical registers removed."""
    stripped = QuantumCircuit(*circuit.qregs, name=circuit.name)
    for instruction in circuit.data:
        if instruction.operation.name in ("measure", "barrier"):
            continue
        stripped.append(instruction.operation, instruction.qubits, [])
    return stripped


def _apply_optimization_passes(circuit: Circuit, optimization_level: int) -> None:
    """Apply pytket optimisation passes scaled by level (Selene path only).

    Level 0: no optimisation (AutoRebase only, applied in caller).
    Level 1: remove redundant gates before rebasing.
    Level 2: full peephole optimisation (CX target, allow_swaps=False) + redundancy removal.
    Level 3: level 2 + single-qubit squash.
    """
    if optimization_level >= 1:
        RemoveRedundancies().apply(circuit)

    if optimization_level >= 2:
        try:
            FullPeepholeOptimise(allow_swaps=False, target_2qb_gate=OpType.CX).apply(circuit)
        except (ValueError, RuntimeError):
            FullPeepholeOptimise(allow_swaps=False).apply(circuit)
        RemoveRedundancies().apply(circuit)

    if optimization_level >= 3:
        SquashTK1().apply(circuit)


def _extract_register_infos(circuit: QuantumCircuit) -> list[RegisterInfo]:
    """Return register metadata sorted lexicographically (matching load_pytket arg order)."""
    infos = [RegisterInfo(name=reg.name, size=reg.size) for reg in circuit.qregs]
    infos.sort(key=lambda r: r.name)
    return infos


# ---------------------------------------------------------------------------
# Selene: Guppy wrapper + local emulator
# ---------------------------------------------------------------------------

def build_and_run_guppy(
    pytket_circuit: Circuit,
    register_infos: list[RegisterInfo],
    measurement_plan: list[RegisterMeasurement],
    config: QuantinuumBackendConfig,
    shots: int,
    verbose: bool = True,
) -> dict[str, int]:
    """Wrap a measurement-free pytket circuit in Guppy and run on Selene.

    Dispatches to a register-layout-specific Guppy ``main()`` builder for
    known HHL layouts (3 or 5 registers), or falls back to a generic
    flat-measure-all wrapper for arbitrary register structures.
    """
    from guppylang import guppy

    uid = uuid.uuid4().hex[:8]
    loaded_func = guppy.load_pytket(f"circuit_{uid}", pytket_circuit)
    sizes = [r.size for r in register_infos]
    n_regs = len(register_infos)
    measured_names = {m.quantum_register_name for m in measurement_plan}

    if n_regs == 3 and set(r.name for r in register_infos) >= {
        "ancilla_flag_register", "b_to_x_register", "qpe_register"
    }:
        guppy_main = _build_measure_x_main(loaded_func, *sizes)
    elif n_regs == 5 and set(r.name for r in register_infos) >= {
        "ancilla_flag_register", "b_to_x_register", "qpe_register",
        "swap_test_ancilla_register", "v_register",
    }:
        guppy_main = _build_swap_test_main(loaded_func, *sizes)
    else:
        guppy_main = _build_flat_main(loaded_func, pytket_circuit.n_qubits)
        raw_result = _run_selene(guppy_main, config, shots, verbose=verbose)
        return _combine_flat_results(
            raw_result, register_infos, measurement_plan,
        )

    raw_result = _run_selene(guppy_main, config, shots, verbose=verbose)
    return _combine_shot_results(raw_result, measurement_plan)


# -- measure_x layout: ancilla_flag_register, b_to_x_register, qpe_register --

def _build_measure_x_main(loaded_func: Any, afr_size: int, btr_size: int, qr_size: int) -> Any:
    """Guppy main for measure_x: measure ancilla + b_to_x, discard qpe."""
    from guppylang import guppy
    from guppylang.std.builtins import result, array, comptime
    from guppylang.std.quantum import qubit, measure_array, discard_array

    @guppy
    def main() -> None:
        afr = array(qubit() for _ in range(comptime(afr_size)))
        btr = array(qubit() for _ in range(comptime(btr_size)))
        qr = array(qubit() for _ in range(comptime(qr_size)))
        loaded_func(afr, btr, qr)
        result("ancilla_flag_result", measure_array(afr))
        result("x_result", measure_array(btr))
        discard_array(qr)

    return main


# -- swap_test layout: ancilla_flag, b_to_x, qpe, swap_test_ancilla, v --

def _build_swap_test_main(
    loaded_func: Any,
    afr_size: int, btr_size: int, qr_size: int,
    star_size: int, vr_size: int,
) -> Any:
    """Guppy main for swap_test: measure ancilla + swap_ancilla, discard rest."""
    from guppylang import guppy
    from guppylang.std.builtins import result, array, comptime
    from guppylang.std.quantum import qubit, measure_array, discard_array

    @guppy
    def main() -> None:
        afr = array(qubit() for _ in range(comptime(afr_size)))
        btr = array(qubit() for _ in range(comptime(btr_size)))
        qr = array(qubit() for _ in range(comptime(qr_size)))
        star = array(qubit() for _ in range(comptime(star_size)))
        vr = array(qubit() for _ in range(comptime(vr_size)))
        loaded_func(afr, btr, qr, star, vr)
        result("ancilla_flag_result", measure_array(afr))
        result("swap_test_result", measure_array(star))
        discard_array(btr)
        discard_array(qr)
        discard_array(vr)

    return main


# -- Generic fallback: flatten all qubits, measure everything --

def _build_flat_main(loaded_func: Any, total_qubits: int) -> Any:
    """Guppy main for arbitrary layouts: measure all qubits as a single array."""
    from guppylang import guppy
    from guppylang.std.builtins import result, array, comptime
    from guppylang.std.quantum import qubit, measure_array

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(comptime(total_qubits)))
        loaded_func(qs)
        result("m", measure_array(qs))

    return main


def _combine_flat_results(
    raw_result: Any,
    register_infos: list[RegisterInfo],
    measurement_plan: list[RegisterMeasurement],
) -> dict[str, int]:
    """Parse flat-measure-all results, slicing out the measured registers.

    The flat qubit array follows pytket's lexicographic register ordering
    (matching ``register_infos``).  We compute each register's offset, then
    extract and concatenate only the measured bits in the order expected by
    the post-processor.
    """
    offsets: dict[str, tuple[int, int]] = {}
    idx = 0
    for ri in register_infos:
        offsets[ri.name] = (idx, idx + ri.size)
        idx += ri.size

    tag_order = list(reversed(measurement_plan))
    counts: Counter[str] = Counter()

    for shot in raw_result:
        entries = shot.as_dict()
        full_bits = entries.get("m", "")
        if isinstance(full_bits, (list, tuple)):
            full_bits = "".join("1" if v else "0" for v in full_bits)

        parts: list[str] = []
        for m in tag_order:
            start, end = offsets[m.quantum_register_name]
            parts.append(full_bits[start:end])
        counts["".join(parts)] += 1

    return dict(counts)


def _run_selene(
    guppy_main: Any,
    config: QuantinuumBackendConfig,
    shots: int,
    verbose: bool = True,
) -> Any:
    """Execute the Guppy program on the local Selene emulator."""
    builder = guppy_main.emulator(n_qubits=config.n_qubits)
    if config.seed is not None:
        builder = builder.with_seed(config.seed)
    builder = builder.with_shots(shots)

    if verbose:
        print(f">>> Running on Selene emulator (n_qubits={config.n_qubits}, shots={shots})")

    return builder.run()


# ---------------------------------------------------------------------------
# Nexus: compile job + execute job
# ---------------------------------------------------------------------------

def run_nexus_pytket(
    pytket_circuit: Circuit,
    config: QuantinuumBackendConfig,
    shots: int,
    optimization_level: int,
    measurement_plan: list[RegisterMeasurement],
    verbose: bool = True,
) -> dict[str, int]:
    """Compile and execute a pytket circuit on Quantinuum Nexus.

    Uploads the (minimally preprocessed) pytket circuit, runs a Nexus
    compile job to apply the device-native compilation pass, then submits
    an execute job and returns a counts dict ready for Post_Processor.

    Args:
        pytket_circuit: Circuit WITH measurements from ``prepare_pytket_circuit_for_nexus``.
        config: Quantinuum backend configuration.
        shots: Number of shots to execute.
        optimization_level: Nexus compile-job optimisation level (clamped to 0–2).
        measurement_plan: Register measurement mapping for result parsing.
        verbose: Whether to print progress messages.
    """
    import qnexus as qnx
    from datetime import datetime

    project = qnx.projects.get_or_create(config.project_name)
    qnx.context.set_active_project(project)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backend_config = _build_nexus_backend_config(config)

    circuit_ref = qnx.circuits.upload(
        circuit=pytket_circuit,
        name=f"qlsa-{config.device_name}-{timestamp}",
    )
    if verbose:
        print(f">>> Uploaded circuit to Nexus (project={config.project_name})")

    compile_level = min(optimization_level, 2)
    compile_job = qnx.start_compile_job(
        programs=[circuit_ref],
        backend_config=backend_config,
        optimisation_level=compile_level,
        name=f"qlsa-compile-{timestamp}",
    )
    if verbose:
        print(f">>> Compiling on {config.device_name} (optimisation_level={compile_level})")
    qnx.jobs.wait_for(compile_job, timeout=None)
    compiled_circuit_ref = qnx.jobs.results(compile_job)[0].get_output()
    if verbose:
        print(">>> Compilation complete")

    execute_job = qnx.start_execute_job(
        programs=[compiled_circuit_ref],
        n_shots=[shots],
        backend_config=backend_config,
        name=f"qlsa-execute-{timestamp}",
    )
    if verbose:
        print(f">>> Submitted execution job on {config.device_name} ({shots} shots)")
    qnx.jobs.wait_for(execute_job, timeout=None)
    result = qnx.jobs.results(execute_job)[0].download_result()
    if verbose:
        print(">>> Execution complete, results downloaded.")

    return _parse_backend_result(result, measurement_plan)


def _build_nexus_backend_config(config: QuantinuumBackendConfig) -> Any:
    """Build the appropriate qnexus backend config for the target device."""
    import qnexus as qnx

    device = config.device_name
    if device.startswith("Helios"):
        helios_kwargs: dict[str, Any] = {
            "system_name": device,
            "emulator_config": qnx.models.HeliosEmulatorConfig(n_qubits=config.n_qubits),
        }
        if config.max_cost is not None:
            helios_kwargs["max_cost"] = config.max_cost
        return qnx.models.HeliosConfig(**helios_kwargs)

    qtm_kwargs: dict[str, Any] = {
        "device_name": device,
        "attempt_batching": True,
    }
    if config.max_batch_cost is not None:
        qtm_kwargs["max_batch_cost"] = config.max_batch_cost
    if config.is_emulator and config.noisy_simulation:
        qtm_kwargs["noisy_simulation"] = True
    return qnx.QuantinuumConfig(**qtm_kwargs)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_backend_result(
    result: Any,
    measurement_plan: list[RegisterMeasurement],
) -> dict[str, int]:
    """Convert a pytket BackendResult into a composite counts dict.

    Assembles classical bits from each register in the same order used by
    ``_combine_shot_results`` (reversed ``measurement_plan``), producing
    bitstrings compatible with Post_Processor.
    """
    from pytket.circuit import Bit

    # Build the cbit list in the same order as _combine_shot_results:
    # - registers in reversed(measurement_plan) order (x_result MSB, ancilla LSB)
    # - within each register, highest index first (MSB-first, matching Qiskit
    #   get_counts() convention where qubit[n-1] is the leftmost bit).
    ordered_bits = [
        Bit(m.classical_register_name, i)
        for m in reversed(measurement_plan)
        for i in range(m.size - 1, -1, -1)
    ]
    raw_counts = result.get_counts(cbits=ordered_bits)
    return {"".join(str(b) for b in bitvals): count for bitvals, count in raw_counts.items()}


def _combine_shot_results(
    raw_result: Any,
    measurement_plan: list[RegisterMeasurement],
) -> dict[str, int]:
    """Convert raw Selene results into a composite counts dict.

    Iterates over individual shots, extracts each measured register's
    bitstring, and concatenates them so that the *last* entry in
    ``measurement_plan`` becomes the leftmost (most-significant) bits and the
    *first* entry becomes the rightmost (least-significant) bit, matching
    Qiskit's ``join_data(names=[...]).get_counts()`` convention.
    """
    tag_order = [m.classical_register_name for m in reversed(measurement_plan)]
    counts: Counter[str] = Counter()

    for shot in raw_result:
        entries = shot.as_dict()
        parts: list[str] = []
        for tag in tag_order:
            val = entries.get(tag, "")
            parts.append(_value_to_bitstring(val))
        counts["".join(parts)] += 1

    return dict(counts)


def _value_to_bitstring(val: Any) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        # Guppy's measure_array returns [qubit[0], qubit[1], ...] (LSB first).
        # Qiskit's get_counts() uses MSB-first (big-endian) convention, so
        # qubit[0] appears at the RIGHTMOST position in the bitstring.
        # Reversing here aligns the two conventions so that
        # int(bitstring, 2) produces the correct coordinate index.
        return "".join("1" if v else "0" for v in reversed(val))
    return str(val)
