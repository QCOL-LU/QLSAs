from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


_VALID_DD_SEQUENCE_TYPES = {"XX", "XpXm", "XY4"}
_VALID_DD_SCHEDULING_METHODS = {"alap", "asap"}


@dataclass(slots=True)
class IBMExecutionOptions:
    """IBM Runtime execution options used by the SamplerV2 backend path."""

    enable_error_mitigation: bool = False
    enable_dynamical_decoupling: bool = False
    dd_sequence_type: str = "XX"
    dd_scheduling_method: Optional[str] = None
    enable_gate_twirling: bool = False
    twirling_num_randomizations: Optional[int] = None
    twirling_shots_per_randomization: Optional[int] = None

    def __post_init__(self) -> None:
        if self.dd_sequence_type not in _VALID_DD_SEQUENCE_TYPES:
            raise ValueError(
                f"Invalid DD sequence type: {self.dd_sequence_type}. "
                f"Must be one of {sorted(_VALID_DD_SEQUENCE_TYPES)}."
            )
        if (
            self.dd_scheduling_method is not None
            and self.dd_scheduling_method not in _VALID_DD_SCHEDULING_METHODS
        ):
            raise ValueError(
                f"Invalid DD scheduling method: {self.dd_scheduling_method}. "
                f"Must be one of {sorted(_VALID_DD_SCHEDULING_METHODS)}."
            )
        if (
            self.twirling_num_randomizations is not None
            and self.twirling_num_randomizations <= 0
        ):
            raise ValueError("twirling_num_randomizations must be positive.")
        if (
            self.twirling_shots_per_randomization is not None
            and self.twirling_shots_per_randomization <= 0
        ):
            raise ValueError("twirling_shots_per_randomization must be positive.")


def apply_ibm_error_mitigation_options(
    sampler_options: Any,
    ibm_options: Optional[IBMExecutionOptions],
) -> None:
    """Apply IBM Runtime mitigation settings onto a sampler options object."""
    if ibm_options is None or not ibm_options.enable_error_mitigation:
        return

    if ibm_options.enable_dynamical_decoupling:
        dd_options = sampler_options.dynamical_decoupling
        dd_options.enable = True
        dd_options.sequence_type = ibm_options.dd_sequence_type
        if ibm_options.dd_scheduling_method is not None:
            dd_options.scheduling_method = ibm_options.dd_scheduling_method

    if ibm_options.enable_gate_twirling:
        twirling_options = sampler_options.twirling
        twirling_options.enable_gates = True
        if ibm_options.twirling_num_randomizations is not None:
            twirling_options.num_randomizations = ibm_options.twirling_num_randomizations
        if ibm_options.twirling_shots_per_randomization is not None:
            twirling_options.shots_per_randomization = (
                ibm_options.twirling_shots_per_randomization
            )
