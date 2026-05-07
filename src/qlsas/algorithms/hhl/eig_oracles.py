"""Eigenvalue-inversion oracle strategies for the HHL algorithm.

Each oracle is a lightweight strategy object that appends the appropriate
rotation circuit to a partially-built HHL circuit.  Pass one to
:class:`~qlsas.algorithms.hhl.hhl.HHL` at construction time::

    from qlsas.algorithms.hhl.eig_oracles import UCRYEigOracle
    hhl = HHL(num_qpe_qubits=4, eig_oracle=UCRYEigOracle())

Available oracles
-----------------
:class:`MCRYEigOracle`
    One ``m``-controlled RY per QPE basis state.  Brute-force lookup; depth
    ``O(m * 2^m)``.
:class:`UCRYEigOracle`
    Möttönen uniformly-controlled RY decomposition.  Depth ``O(2^m)``,
    same unitary as :class:`MCRYEigOracle`.  Default oracle on :class:`HHL`.
:class:`ExactReciprocalEigOracle`
    Wraps Qiskit's :class:`ExactReciprocalGate` (which itself decomposes to
    a UCRY tree).  Same asymptotic depth as :class:`UCRYEigOracle` with a
    smaller constant factor, at the cost of saturation and boundary-state
    quirks — see ``docs/eigenvalue_inversion.md`` for the trade-offs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qlsas.algorithms.hhl.hhl_helpers import (
    mcry_eig_inversion,
    ucry_eig_inversion,
    exact_reciprocal_eig_inversion,
)


class EigOracle(ABC):
    """Abstract strategy for eigenvalue inversion in HHL.

    Implementations receive the partially-built circuit, the QPE and ancilla
    registers, and the problem parameters (*A*, *t0*, *C*), then append the
    appropriate rotation gates in-place.
    """

    @abstractmethod
    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        """Append eigenvalue-inversion gates to *circ*.

        Parameters
        ----------
        circ : QuantumCircuit
            The circuit being built (mutated in place).
        qpe_register : QuantumRegister
            The QPE register whose state encodes the eigenvalue estimate.
        ancilla_qubit :
            The single ancilla qubit to rotate.
        A : np.ndarray
            The Hermitian coefficient matrix.
        t0 : float
            Hamiltonian evolution time (determines the QPE phase scaling).
        C : float
            Ancilla rotation scaling factor.
        """
        ...


class MCRYEigOracle(EigOracle):
    """Eigenvalue inversion via one ``m``-controlled RY per QPE basis state.

    For every integer ``k`` in ``[1, 2^m)`` the oracle appends a single
    multi-controlled RY gate whose control state is ``|k⟩`` and whose
    rotation angle is ``2·arcsin(C / λ_k)``, where ``λ_k`` is the eigenvalue
    implied by the QPE phase ``k / 2^m``.  Decomposed depth scales as
    ``O(m · 2^m)`` because each MCRY decomposes to roughly ``m`` Toffoli /
    CNOT gates.

    Same rotation table (and therefore the same unitary on populated states)
    as :class:`UCRYEigOracle`; choose this oracle only if you need the
    explicit per-basis-state structure for debugging or transpilation.
    """

    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        mcry_eig_inversion(circ, qpe_register, ancilla_qubit, A=A, t0=t0, C=C)


class UCRYEigOracle(EigOracle):
    """Eigenvalue inversion via a Möttönen uniformly-controlled RY tree.

    Computes the same rotation-angle table as :class:`MCRYEigOracle` but
    realises it as a recursive ``CNOT + RY`` tree (Möttönen et al., 2004).
    Decomposed depth ``O(2^m)``, gate count ``2^m`` RY + ``2^(m+1) − 2``
    CX, no ancillas.  Identical unitary to :class:`MCRYEigOracle` on every
    QPE basis state.

    Recommended default — same correctness as :class:`MCRYEigOracle` with
    an asymptotic factor-``m`` depth saving and no edge-case footguns.
    """

    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        ucry_eig_inversion(circ, qpe_register, ancilla_qubit, A=A, t0=t0, C=C)


class ExactReciprocalEigOracle(EigOracle):
    """Eigenvalue inversion via Qiskit's :class:`ExactReciprocalGate`.

    Same asymptotic depth as :class:`UCRYEigOracle` (the gate decomposes
    to a UCRY internally) and ~10–20 % smaller constant factor.  Two
    structural caveats:

    * **Saturation drop.** When ``|S · nl / i| > 1`` the gate emits a
      zero rotation instead of clamping to ``π`` — leakage amplitude on
      saturating QPE states is lost.
    * **Boundary-state hole.** When ``neg_vals=True`` the gate hard-codes
      the rotation on QPE state ``|2^(m-1)⟩`` (the most-negative phase
      in two's complement) to zero, regardless of the spectrum.

    Prefer :class:`UCRYEigOracle` unless you specifically want the
    Qiskit-supplied decomposition; see ``docs/eigenvalue_inversion.md`` for
    when the saturation and boundary issues matter in practice.
    """

    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        exact_reciprocal_eig_inversion(
            circ, qpe_register, ancilla_qubit, A=A, t0=t0, C=C
        )
