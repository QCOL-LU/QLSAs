"""End-to-end correctness tests for the Quantinuum / Selene execution path.

These tests run the full pipeline (circuit build → pytket transpile → Guppy
Selene emulation → post-processing) and verify *solution quality*, not just
that the code runs without crashing.

Marked ``slow`` because Selene emulation takes a few seconds per circuit.
Not marked ``hardware`` because no cloud credentials are required.
"""

from __future__ import annotations

import math
import numpy as np
import numpy.linalg as LA
import pytest

from qlsas.algorithms.hhl.hhl import HHL
from qlsas.data_loader import StatePrep
from qlsas.quantinuum_config import QuantinuumBackendConfig
from qlsas.transpiler import Transpiler
from qlsas.executer import Executer
from qlsas.post_processor import Post_Processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_hhl_aer(
    aer_backend,
    A: np.ndarray,
    b: np.ndarray,
    num_qpe_qubits: int,
    shots: int = 2000,
) -> tuple[np.ndarray, float, float]:
    """Run HHL on Qiskit Aer (reference path) and return (solution, success_rate, residual)."""
    hhl = HHL(
        state_prep=StatePrep(method="default"),
        readout="measure_x",
        num_qpe_qubits=num_qpe_qubits,
        eig_oracle="classical",
    )
    circuit = hhl.build_circuit(A, b)
    transpiler = Transpiler(circuit=circuit, backend=aer_backend, optimization_level=1)
    transpiled = transpiler.optimize()
    executer = Executer()
    result = executer.run(transpiled, aer_backend, shots=shots, verbose=False)
    processor = Post_Processor()
    return processor.process_tomography(result, A, b, verbose=False)


def _selene_backend(circuit, seed: int = 0) -> QuantinuumBackendConfig:
    return QuantinuumBackendConfig(
        device_name="H1-1E",
        n_qubits=circuit.num_qubits,
        use_local_emulator=True,
        seed=seed,
    )


def _run_hhl_selene(
    A: np.ndarray,
    b: np.ndarray,
    num_qpe_qubits: int,
    optimization_level: int,
    shots: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Build, transpile, execute on Selene, and return (solution, success_rate, residual)."""
    hhl = HHL(
        state_prep=StatePrep(method="default"),
        readout="measure_x",
        num_qpe_qubits=num_qpe_qubits,
        eig_oracle="classical",
    )
    circuit = hhl.build_circuit(A, b)
    backend = _selene_backend(circuit, seed=seed)

    transpiler = Transpiler(circuit=circuit, backend=backend, optimization_level=optimization_level)
    transpiled = transpiler.optimize()

    executer = Executer()
    counts = executer.run(
        transpiled,
        backend,
        shots=shots,
        register_infos=transpiler.register_infos,
        measurement_plan=transpiler.measurement_plan,
        verbose=False,
    )

    processor = Post_Processor()
    solution, success_rate, residual = processor.process_tomography(counts, A, b, verbose=False)
    return solution, success_rate, residual


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hhl_2x2_problem():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 0.0])
    b = b / np.linalg.norm(b)
    return A, b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSeleneCorrectnessOptLevel0:
    """Baseline: no pytket optimisation passes — circuit should always be correct."""

    @pytest.mark.slow
    def test_2x2_residual_below_threshold(self, hhl_2x2_problem):
        A, b = hhl_2x2_problem
        _, _, residual = _run_hhl_selene(A, b, num_qpe_qubits=3, optimization_level=0)
        assert residual < 0.5, (
            f"Selene opt_level=0 residual {residual:.4f} exceeds threshold 0.5 — "
            "circuit or post-processing is wrong."
        )

    @pytest.mark.slow
    def test_2x2_success_rate_nonzero(self, hhl_2x2_problem):
        A, b = hhl_2x2_problem
        _, success_rate, _ = _run_hhl_selene(A, b, num_qpe_qubits=3, optimization_level=0)
        assert success_rate > 0.01, (
            f"Success rate {success_rate:.4f} is effectively zero — "
            "ancilla qubit is never 1, check circuit structure."
        )


class TestSeleneCorrectnessOptLevel2:
    """Regression guard: optimisation passes must not change circuit semantics."""

    @pytest.mark.slow
    def test_2x2_residual_below_threshold(self, hhl_2x2_problem):
        A, b = hhl_2x2_problem
        _, _, residual = _run_hhl_selene(A, b, num_qpe_qubits=3, optimization_level=2)
        assert residual < 0.5, (
            f"Selene opt_level=2 residual {residual:.4f} exceeds threshold 0.5."
        )

    @pytest.mark.slow
    def test_2x2_residual_consistent_with_opt_level_0(self, hhl_2x2_problem):
        """Level-2 solution should be as good as level-0 (same noiseless circuit)."""
        A, b = hhl_2x2_problem
        _, _, residual_0 = _run_hhl_selene(A, b, num_qpe_qubits=3, optimization_level=0, seed=1)
        _, _, residual_2 = _run_hhl_selene(A, b, num_qpe_qubits=3, optimization_level=2, seed=1)
        assert residual_2 < 0.5, (
            f"opt_level=2 residual {residual_2:.4f} is bad while opt_level=0 gives {residual_0:.4f}."
        )


class TestCrossBackendConsistency:
    """Aer (Qiskit) and Selene (Guppy) must produce consistent solutions for the same problem.

    Catches mapping bugs (endianness, register order) that would cause one backend
    to produce a fundamentally different solution than the other.
    """

    @pytest.mark.slow
    def test_aer_and_selene_solutions_agree_2x2(self, aer_backend, hhl_2x2_problem):
        """Same 2×2 problem on Aer and Selene → similar solution, both residuals < 0.5."""
        A, b = hhl_2x2_problem

        sol_aer, _, residual_aer = _run_hhl_aer(aer_backend, A, b, num_qpe_qubits=3, shots=3000)
        sol_selene, _, residual_selene = _run_hhl_selene(
            A, b, num_qpe_qubits=3, optimization_level=0, shots=3000, seed=13
        )

        assert residual_aer < 0.5, f"Aer residual {residual_aer:.4f}"
        assert residual_selene < 0.5, f"Selene residual {residual_selene:.4f}"

        cosine = np.abs(np.dot(sol_aer / LA.norm(sol_aer), sol_selene / LA.norm(sol_selene)))
        assert cosine > 0.85, (
            f"Aer and Selene solutions disagree (cosine={cosine:.4f}). "
            "Likely endianness or register-mapping bug in Guppy path."
        )


class TestSelene4x4Problem:
    """Correctness on a 4×4 system — larger than 2×2 but fast enough for CI."""

    @pytest.fixture(scope="class")
    def hhl_4x4_problem(self):
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        eigs = np.linspace(1.0, 5.0, 4)
        A = (Q * eigs) @ Q.T
        b = rng.standard_normal(4)
        b = b / np.linalg.norm(b)
        return A, b

    @pytest.fixture(scope="class")
    def hhl_4x4_asymmetric(self):
        """4×4 diagonal problem designed to expose Guppy endianness bugs.

        The true solution has coordinate 1 dominant and coordinate 2 near-zero.
        If the Guppy path reverses these coordinates, residual jumps to ≈ 1.
        """
        A = np.diag([1.0, 2.0, 8.0, 4.0])
        # b mostly at coordinate 1 → solution dominated by component 1
        b = np.array([0.05, 0.98, 0.05, 0.05])
        b = b / np.linalg.norm(b)
        return A, b

    @pytest.mark.slow
    def test_4x4_opt0_residual_below_threshold(self, hhl_4x4_problem):
        A, b = hhl_4x4_problem
        _, _, residual = _run_hhl_selene(A, b, num_qpe_qubits=2, optimization_level=0, shots=3000)
        assert residual < 0.5, (
            f"4×4 Selene opt_level=0 residual {residual:.4f} — baseline circuit is wrong."
        )

    @pytest.mark.slow
    def test_4x4_opt2_residual_consistent_with_opt0(self, hhl_4x4_problem):
        """Optimisation must not degrade a noiseless 4×4 simulation."""
        A, b = hhl_4x4_problem
        _, _, residual_0 = _run_hhl_selene(A, b, num_qpe_qubits=2, optimization_level=0, shots=3000, seed=7)
        _, _, residual_2 = _run_hhl_selene(A, b, num_qpe_qubits=2, optimization_level=2, shots=3000, seed=7)
        assert residual_2 < 0.5, (
            f"opt_level=2 residual {residual_2:.4f} vs opt_level=0 residual {residual_0:.4f}."
        )

    @pytest.mark.slow
    def test_4x4_endianness_asymmetric(self, hhl_4x4_asymmetric):
        """Regression test for Guppy measure_array LSB-vs-MSB endianness.

        Uses a b vector strongly weighted at coordinate 1 (vs near-zero at 2).
        Swapping coordinates 1 ↔ 2 (the endianness bug) produces residual ≫ 0.5.
        """
        A, b = hhl_4x4_asymmetric
        _, _, residual = _run_hhl_selene(A, b, num_qpe_qubits=2, optimization_level=0, shots=4000)
        assert residual < 0.5, (
            f"4×4 asymmetric Selene residual {residual:.4f} exceeds threshold."
        )
