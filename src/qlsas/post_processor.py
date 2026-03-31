import numpy as np
import math
import numpy.linalg as LA
from qiskit.primitives.containers import SamplerPubResult


class Post_Processor:
    """Post-processing for quantum linear solver results.

    Supports two result types:
      - ``SamplerPubResult`` from Qiskit/IBM backends.
      - ``dict[str, int]`` counts dicts from the Quantinuum/Guppy path.
    """

    def process_tomography(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """Process the result using tomography.

        Returns (solution_vector, success_rate, residual).
        """
        if isinstance(result, SamplerPubResult):
            return self.process_qiskit_tomography(result, A, b, verbose=verbose)
        elif isinstance(result, dict):
            return self.process_quantinuum_tomography(result, A, b, verbose=verbose)
        else:
            raise ValueError(f"Invalid result type: {type(result)}.")

    def process_swap_test(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
        swap_test_vector: np.ndarray,
    ) -> tuple[float, float, float]:
        """Process the result using the swap test.

        Returns (expected_value, success_rate, residual).
        """
        if isinstance(result, SamplerPubResult):
            return self.process_qiskit_swap_test(result, A, b, swap_test_vector)
        elif isinstance(result, dict):
            return self.swap_test_from_counts(result, A, b, swap_test_vector)
        else:
            raise ValueError(f"Invalid result type: {type(result)}.")

    # ------------------------------------------------------------------
    # Backend-agnostic core (operates on counts dicts)
    # ------------------------------------------------------------------

    def norm_estimation(self, A, b, x):
        """Estimate the norm of x such that ||Ax - b|| is minimized."""
        v = A @ x
        denominator = np.dot(v, v)
        if denominator == 0:
            return 1e-10
        return np.dot(v, b) / denominator

    def tomography_from_counts(
        self,
        counts: dict[str, int],
        A: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Run tomography using a counts dict.

        Keys are bitstrings: x_result bits + ancilla flag; success when
        last bit is '1'.
        Returns (solution, success_rate, residual).
        """
        x_size = int(math.log2(len(b)))
        num_successful_shots = 0
        approximate_solution = np.zeros(len(b))
        total_shots = sum(counts.values())

        for key, value in counts.items():
            if key[-1] == '1':
                num_successful_shots += value
                coord = int(key[:x_size], base=2)
                approximate_solution[coord] = value
        if num_successful_shots == 0:
            raise ValueError("No successful shots.")
        approximate_solution = np.sqrt(approximate_solution / num_successful_shots)
        success_rate = num_successful_shots / total_shots if total_shots else 0.0
        return self._finish_tomography(
            approximate_solution, success_rate, num_successful_shots, total_shots, A, b
        )

    def swap_test_from_counts(
        self,
        counts: dict[str, int],
        A: np.ndarray,
        b: np.ndarray,
        swap_test_vector: np.ndarray,
    ) -> tuple[float, float, float]:
        """Run swap-test post-processing using a counts dict.

        Keys are bitstrings: swap_test_bit + ancilla_flag_bit; success when
        last bit is '1'.
        Returns (expected_value, success_rate, residual).
        """
        correct_shots = 0
        num_swap_ones = 0
        total_shots = sum(counts.values())

        for key, value in counts.items():
            if key[-1] == '1':
                correct_shots += value
                if key[0] == '1':
                    num_swap_ones += value

        if total_shots == 0:
            raise ValueError("No shots recorded.")
        if correct_shots == 0:
            raise ValueError("No successful HHL shots.")

        success_rate = correct_shots / total_shots
        exp_value = num_swap_ones / correct_shots

        classical_solution = LA.solve(A, b)
        if LA.norm(classical_solution) > 0:
            normalized_classical = classical_solution / LA.norm(classical_solution)
        else:
            normalized_classical = classical_solution

        if LA.norm(swap_test_vector) > 0:
            normalized_swap = swap_test_vector / LA.norm(swap_test_vector)
        else:
            normalized_swap = swap_test_vector

        overlap = np.vdot(normalized_swap, normalized_classical)
        expected_prob = 0.5 - 0.5 * (np.abs(overlap) ** 2)
        residual = abs(exp_value - expected_prob)
        return exp_value, success_rate, residual

    def _finish_tomography(
        self,
        approximate_solution: np.ndarray,
        success_rate: float,
        num_successful_shots: int,
        total_shots: int,
        A: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Apply sign correction, normalization, and residual.

        Returns (solution, success_rate, residual).
        """
        classical_solution = LA.solve(A, b)
        for i in range(len(approximate_solution)):
            approximate_solution[i] = approximate_solution[i] * np.sign(classical_solution[i])

        assert np.allclose(
            sum(approximate_solution[i] ** 2 for i in range(len(approximate_solution))),
            1.0,
            atol=1e-6,
        ), "Approximate solution is not normalized."

        scaling_factor = self.norm_estimation(A, b, approximate_solution)
        scaled_solution = approximate_solution * scaling_factor
        residual = np.linalg.norm(b - A @ scaled_solution)
        return approximate_solution, success_rate, residual

    # ------------------------------------------------------------------
    # Qiskit-specific (SamplerPubResult)
    # ------------------------------------------------------------------

    def process_qiskit_tomography(
        self,
        result: SamplerPubResult,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        counts = result.join_data(names=['ancilla_flag_result', 'x_result']).get_counts()

        total_shots = sum(counts.values())
        approximate_solution, success_rate, residual = self.tomography_from_counts(counts, A, b)
        num_successful_shots = sum(v for k, v in counts.items() if k[-1] == '1')

        if verbose:
            print(f"total shots: {total_shots}")
            print(f"num_successful_shots: {num_successful_shots}")
            print(f"success rate: {success_rate}")
            print(f"solver residual: {residual}")
        return approximate_solution, success_rate, residual

    def process_qiskit_swap_test(
        self,
        result: SamplerPubResult,
        A: np.ndarray,
        b: np.ndarray,
        swap_test_vector: np.ndarray,
    ) -> tuple[float, float, float]:
        counts = result.join_data(names=['ancilla_flag_result', 'swap_test_result']).get_counts()
        return self.swap_test_from_counts(counts, A, b, swap_test_vector)

    # ------------------------------------------------------------------
    # Quantinuum-specific (dict counts)
    # ------------------------------------------------------------------

    def process_quantinuum_tomography(
        self,
        counts: dict[str, int],
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """Process Quantinuum counts dict for tomography."""
        total_shots = sum(counts.values())
        approximate_solution, success_rate, residual = self.tomography_from_counts(counts, A, b)
        num_successful_shots = sum(v for k, v in counts.items() if k[-1] == '1')

        if verbose:
            print(f"total shots: {total_shots}")
            print(f"num_successful_shots: {num_successful_shots}")
            print(f"success rate: {success_rate}")
            print(f"solver residual: {residual}")
        return approximate_solution, success_rate, residual
