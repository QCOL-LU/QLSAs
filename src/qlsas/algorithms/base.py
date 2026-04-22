from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qlsas.readout.base import QLSACircuit


class QLSA(ABC):
    """Abstract base for a Quantum Linear Systems Algorithm.

    Implementations build the *core* quantum circuit (state preparation,
    algorithm body, uncomputation) and return a :class:`QLSACircuit` that
    carries register metadata.  Readout is handled separately by a
    :class:`~qlsas.readout.base.Readout` strategy.
    """

    @abstractmethod
    def build_circuit(
        self,
        A: np.ndarray,
        b: np.ndarray,
        state_prep,
        **kwargs,
    ) -> QLSACircuit:
        """Build the core QLSA circuit (no readout measurements).

        Parameters
        ----------
        A : np.ndarray
            Hermitian matrix of the linear system.
        b : np.ndarray
            Unit-norm right-hand-side vector.
        state_prep : StatePrep
            Strategy for loading ``|b>`` into the circuit.

        Returns
        -------
        QLSACircuit
            The circuit plus register metadata for downstream readout.
        """
        raise NotImplementedError
