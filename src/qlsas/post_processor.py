import numpy as np
import math
import numpy.linalg as LA
from qiskit.primitives.containers import SamplerPubResult

class Post_Processor:
    """
    Post_Processor class for post-processing the result of the quantum linear solver.
    """

    def process_tomography( # TODO: add support for quantinuum result
        self, 
        result,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True
    ) -> tuple[np.ndarray, float]:
        """
        Process the result of the quantum linear solver using tomography and return the solution vector.
        Args:
            result: The result of the quantum linear solver.
            A: The matrix representing the linear system.
            b: The vector representing the right-hand side of the linear system.
            verbose: Whether to print results.
        Returns:
            Tuple of (solution vector, success rate, residual).
        """
        if isinstance(result, SamplerPubResult):
            return self.process_qiskit_tomography(result, A, b, verbose=verbose)
        else:
            raise ValueError(f"Invalid result type: {type(result)}.  Quantinuum result not yet supported.")

    
    def process_swap_test(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
        swap_test_vector: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Process the result of the quantum linear solver using swap test and return 
        the expected value of the swap test, the success rate, and the residual.
        Args:
            result: The result of the quantum linear solver.
            A: The matrix representing the linear system.
            b: The vector representing the right-hand side of the linear system.
            swap_test_vector: The vector to use for the swap test.
        Returns:
            The expected value of the swap test, the success rate, and the residual.
        """
        if isinstance(result, SamplerPubResult):
            return self.process_qiskit_swap_test(result, A, b, swap_test_vector)
        else:
            raise ValueError(f"Invalid result type: {type(result)}.  Quantinuum result not yet supported.")

    
    def norm_estimation(self, A, b, x):
        """
        Estimate the norm of the solution x such that ||Ax - b|| is minimized.
        """
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
        """
        Run tomography using a counts dict (e.g. merged across batches).
        Keys are bitstrings: x_result bits + ancilla flag; success when last bit is '1'.
        Returns (solution, success_rate, residual).
        """
        # extract amplitudes of x register
        x_size = int(math.log2(len(b)))
        num_successful_shots = 0  # for normalization
        approximate_solution = np.zeros(len(b))
        total_shots = sum(counts.values())

        for key, value in counts.items():
            if key[-1] == '1':  # ancilla measurement successful
                num_successful_shots += value
                coord = int(key[:x_size], base=2)  # position in b vector from binary string
                approximate_solution[coord] = value
        if num_successful_shots == 0:
            raise ValueError("No successful shots.")
        approximate_solution = np.sqrt(approximate_solution / num_successful_shots)
        # calculate success rate
        success_rate = num_successful_shots / total_shots if total_shots else 0.0
        return self._finish_tomography(approximate_solution, success_rate, num_successful_shots, total_shots, A, b)

    def _finish_tomography(
        self,
        approximate_solution: np.ndarray,
        success_rate: float,
        num_successful_shots: int,
        total_shots: int,
        A: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """
        Apply sign correction, normalization, and residual using pre-aggregated solution.
        Returns (solution, success_rate, residual).
        """
        # extract signs of each solution coordinate, using classical solution for now (to be updated)
        classical_solution = LA.solve(A, b)
        for i in range(len(approximate_solution)):
            approximate_solution[i] = approximate_solution[i] * np.sign(classical_solution[i])

        assert np.allclose(
            sum(approximate_solution[i] ** 2 for i in range(len(approximate_solution))),
            1.0,
            atol=1e-6,
        ), "Approximate solution is not normalized."

        # Scale the normalized solution to minimize the residual
        scaling_factor = self.norm_estimation(A, b, approximate_solution)
        scaled_solution = approximate_solution * scaling_factor
        residual = np.linalg.norm(b - A @ scaled_solution)
        return approximate_solution, success_rate, residual

    def process_qiskit_tomography(
        self,
        result: SamplerPubResult,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True
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
        """
        Process the result of the quantum linear solver using swap test and return 
        the expected value of the swap test, the success rate, and the residual.
        Args:
            result: The result of the quantum linear solver.
            A: The matrix representing the linear system.
            b: The vector representing the right-hand side of the linear system.
            swap_test_vector: The vector to use for the swap test.
        Returns:
            The expected value of the swap test, the success rate, and the residual.
        """
        correct_shots = 0
        num_swap_ones = 0

        counts = result.join_data(names=['ancilla_flag_result', 'swap_test_result']).get_counts()
        total_shots = sum(counts.values())
        for key, value in counts.items():
            # postprocess shots to only consider those with ancilla measurement '1'
            if key[-1] == '1':
                correct_shots += value
                # among correct shots, count how many times the swap ancilla measurement is '1'
                if key[0] == '1':
                    num_swap_ones += value

        if total_shots == 0:
            raise ValueError("No shots recorded.")
        if correct_shots == 0:
            raise ValueError("No successful HHL shots.")

        success_rate = correct_shots / total_shots
        exp_value = num_swap_ones / correct_shots

        # Calculate classical solution for reference
        classical_solution = LA.solve(A, b)
        if LA.norm(classical_solution) > 0:
            normalized_classical = classical_solution / LA.norm(classical_solution)
        else:
            normalized_classical = classical_solution

        # Normalize swap vector if not already
        if LA.norm(swap_test_vector) > 0:
            normalized_swap = swap_test_vector / LA.norm(swap_test_vector)
        else:
            normalized_swap = swap_test_vector

        # Calculate expected overlap
        # P(1) = 0.5 - 0.5 * |<swap|x>|^2
        overlap = np.vdot(normalized_swap, normalized_classical)
        expected_prob = 0.5 - 0.5 * (np.abs(overlap) ** 2)

        residual = abs(exp_value - expected_prob)
        return exp_value, success_rate, residual
