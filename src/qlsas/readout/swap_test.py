"""SwapTest readout: estimate inner product between solution and a reference vector."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister

from qlsas.readout.base import Readout, QLSACircuit
from qlsas.post_processor import Post_Processor


class SwapTestReadout(Readout):
    """Append a swap-test circuit that estimates ⟨x|v⟩.

    Parameters
    ----------
    swap_test_vector : np.ndarray
        The reference vector *v* to compare against the solution.
    state_prep : StatePrep, optional
        A ``StatePrep``-like object whose ``load_state(v)`` method returns a
        ``QuantumCircuit`` that prepares |v⟩.  If *None*, a *state_prep* must
        be provided when calling :meth:`apply`.
    post_processor : Post_Processor, optional
        Custom post-processor instance.  Defaults to a fresh
        :class:`~qlsas.post_processor.Post_Processor`.
    """

    # Classical registers joined for post-processing (ancilla flag last).
    _REGISTER_NAMES: list[str] = ["ancilla_flag_result", "swap_test_result"]

    def __init__(
        self,
        swap_test_vector: np.ndarray,
        state_prep=None,
        post_processor: Post_Processor | None = None,
    ) -> None:
        self.swap_test_vector = swap_test_vector
        self._default_state_prep = state_prep
        self._pp = post_processor or Post_Processor()

    # ------------------------------------------------------------------
    # Readout interface
    # ------------------------------------------------------------------

    @property
    def register_names(self) -> list[str]:
        return self._REGISTER_NAMES

    def apply(
        self,
        qlsa_circuit: QLSACircuit,
        *,
        state_prep=None,
    ):
        """Append the swap-test circuit to *qlsa_circuit*.

        Parameters
        ----------
        qlsa_circuit : QLSACircuit
            Core HHL circuit (no readout yet).
        state_prep : StatePrep, optional
            Used to load the reference vector |v⟩.  Overrides the
            *state_prep* given at construction time.  At least one of the
            two must be provided.
        """
        effective_sp = state_prep or self._default_state_prep
        if effective_sp is None:
            raise ValueError(
                "SwapTestReadout requires a state_prep to load the reference "
                "vector.  Either pass one at construction time or supply one "
                "via the state_prep keyword argument of apply()."
            )

        circ = qlsa_circuit.circuit.copy()
        sol_reg = qlsa_circuit.solution_register

        swap_test_ancilla = QuantumRegister(1, name="swap_test_ancilla_register")
        v_register = QuantumRegister(len(sol_reg), name="v_register")
        swap_test_result = ClassicalRegister(1, name="swap_test_result")
        circ.add_register(swap_test_ancilla, v_register, swap_test_result)

        # Load |v⟩
        circ.compose(
            effective_sp.load_state(self.swap_test_vector),
            v_register,
            inplace=True,
        )
        circ.barrier()

        # Swap test
        circ.h(swap_test_ancilla)
        for i in range(len(sol_reg)):
            circ.cswap(swap_test_ancilla[0], sol_reg[i], v_register[i])
        circ.h(swap_test_ancilla)
        circ.barrier()

        circ.measure(swap_test_ancilla, swap_test_result)
        return circ

    def process(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> tuple[float, float, float]:
        """Compute the swap-test expected value from measurement counts.

        Parameters
        ----------
        result : MeasurementResult or dict
            Wrapped measurement result.
        """
        counts = _to_counts(result, self._REGISTER_NAMES)
        return self._pp.swap_test_from_counts(counts, A, b, self.swap_test_vector)


# ---------------------------------------------------------------------------
# Module-private helper
# ---------------------------------------------------------------------------

def _to_counts(result, register_names: list[str]) -> dict[str, int]:
    """Extract a plain ``dict[str, int]`` from a *result* of any supported type."""
    if isinstance(result, dict):
        return result
    return result.get_counts(register_names)
