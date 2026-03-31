from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_KNOWN_DEVICES = {
    "Helios-1E", "Helios-1",
    "H1-1E", "H1-1",
    "H2-1E", "H2-1", "H2-2E", "H2-2",
}


@dataclass(slots=True)
class QuantinuumBackendConfig:
    """Configuration for Quantinuum backends accessed via Guppy / Nexus.

    Supports two execution modes:
      - **Local emulation** (``use_local_emulator=True``): runs on the Selene
        emulator bundled with guppylang.  No cloud credentials required.
      - **Cloud execution** (``use_local_emulator=False``): compiles to HUGR,
        uploads to Quantinuum Nexus, and executes on the specified device
        (hardware or cloud-hosted emulator).
    """

    device_name: str
    n_qubits: int
    use_local_emulator: bool = False
    project_name: str = "QLSAs"
    max_cost: Optional[int] = None
    noisy_simulation: bool = True
    seed: Optional[int] = None
    max_batch_cost: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValueError("max_cost must be positive when set.")
        if self.max_batch_cost is not None and self.max_batch_cost <= 0:
            raise ValueError("max_batch_cost must be positive when set.")

    @property
    def name(self) -> str:
        """Backend label used in plots and logs."""
        suffix = " (local)" if self.use_local_emulator else ""
        return f"{self.device_name}{suffix}"

    @property
    def is_emulator(self) -> bool:
        return self.device_name.endswith("E")
