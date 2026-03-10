from typing import Union

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend
from qiskit_aer import AerSimulator
from qiskit.providers.backend import BackendV2
from qiskit.transpiler import generate_preset_pass_manager
from pytket.circuit import Circuit
from pytket.extensions.qiskit import tk_to_qiskit

from qlsas.quantinuum_config import QuantinuumBackendConfig
from qlsas.guppy_runner import (
    RegisterInfo,
    RegisterMeasurement,
    extract_measurement_plan,
    prepare_pytket_circuit,
    prepare_pytket_circuit_for_nexus,
)


class Transpiler:
    """Transpiler class for optimizing quantum circuits for target hardware.

    Supports Qiskit/IBM backends and Quantinuum backends (via pytket/Guppy).
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Circuit],
        backend: Union[BackendV2, QuantinuumBackendConfig],
        optimization_level: int,
    ):
        self.circuit = circuit
        self.backend = backend
        self.optimization_level = optimization_level
        self.register_infos: list[RegisterInfo] = []
        self.measurement_plan: list[RegisterMeasurement] = []

    def optimize(self) -> Union[QuantumCircuit, Circuit]:
        """Optimize the circuit for target hardware."""
        if isinstance(self.backend, (BackendV2, IBMBackend, AerSimulator)):
            return self.optimize_qiskit()
        elif isinstance(self.backend, QuantinuumBackendConfig):
            return self.optimize_quantinuum()
        else:
            raise ValueError(f"Invalid backend type: {type(self.backend)}")

    def optimize_qiskit(self) -> QuantumCircuit:
        """Optimize the circuit for IBM hardware."""
        if self.optimization_level not in [0, 1, 2, 3]:
            raise ValueError(
                f"Invalid optimization level: {self.optimization_level}. "
                "Must be 0, 1, 2, or 3."
            )

        if isinstance(self.circuit, Circuit):
            self.circuit = tk_to_qiskit(self.circuit)
        elif not isinstance(self.circuit, QuantumCircuit):
            raise ValueError(
                f"Invalid circuit type: {type(self.circuit)}. "
                "Must be a qiskit QuantumCircuit or a pytket Circuit."
            )

        pm = generate_preset_pass_manager(
            optimization_level=self.optimization_level, backend=self.backend
        )
        return pm.run(self.circuit)

    def optimize_quantinuum(self) -> Circuit:
        """Optimize the circuit for Quantinuum hardware via the pytket bridge.

        Branches on ``backend.use_local_emulator``:

        - **Local Selene**: strips measurements, applies local pytket
          optimisation passes (``FullPeepholeOptimise(CX)`` + ``AutoRebase``),
          and returns a measurement-free circuit for the Guppy wrapper.
        - **Nexus cloud**: keeps measurements, applies only minimal
          preprocessing (Aer decomposition + ``DecomposeBoxes``), and returns
          a circuit ready for ``qnx.start_compile_job``.

        Side effects:
          - Populates ``self.register_infos`` with per-register metadata
            (lexicographic order, matching ``load_pytket`` argument order).
          - Populates ``self.measurement_plan`` with the register-to-classical
            measurement mapping.
        """
        if isinstance(self.circuit, Circuit):
            raise TypeError(
                "optimize_quantinuum expects a Qiskit QuantumCircuit as input, "
                f"got pytket Circuit. Convert with tk_to_qiskit first."
            )

        self.measurement_plan = extract_measurement_plan(self.circuit)

        if self.backend.use_local_emulator:
            pytket_circuit, self.register_infos = prepare_pytket_circuit(
                self.circuit, optimization_level=self.optimization_level
            )
        else:
            pytket_circuit, self.register_infos = prepare_pytket_circuit_for_nexus(
                self.circuit
            )

        return pytket_circuit
