import cudaq
import numpy as np
import math
import numpy.linalg as LA

from typing import Optional

class Post_Processor:
    """
    Post_Processor class for post-processing the result of the quantum linear solver.
    """

    def process(
        self, 
        result,
        readout: str,
        A: np.ndarray,
        b: np.ndarray,
        swap_test_vector: Optional[np.ndarray],
        ):
        """
        Process the result of the quantum linear solver and return the solution vector.
        Args:
            items: The counts dictionary consisting of measured bitstrings and their measurement counts.
        Returns:
            The normalized classical solution vector.
        """
        if readout == "measure_x":
            return self.process_tomography(result, A, b)
        
        elif readout == "swap_test" and swap_test_vector is not None:
            return self.process_swap_test(result, A, b, swap_test_vector)


    def process_tomography(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:

        # extract amplitudes of x register
        x_size = int(math.log2(len(b)))
        num_successful_shots = 0 # for normalization
        approximate_solution = np.zeros(len(b))

        for key, value in result.items():
            if key[0] == '1': # ancilla measurement successful
                num_successful_shots += value
                coord = int(key[-1:0:-1], base=2) # position in b vector from binary string
                approximate_solution[coord] = value
        if num_successful_shots == 0:
            raise ValueError("No successful shots.")
        else:
            approximate_solution = np.sqrt(approximate_solution/num_successful_shots)

        # Calculate success rate
        total_shots = sum(result.values())
        success_rate = num_successful_shots / total_shots if total_shots else 0.0
        
        # extract signs of each solution coordinate, using classical solution for now (to be updated)
        classical_solution = LA.solve(A / LA.norm(b), b / LA.norm(b))
       
        for i in range(len(approximate_solution)):
            approximate_solution[i] = approximate_solution[i] * np.sign(classical_solution[i])
        
        assert np.allclose(
            sum(approximate_solution[i]**2 for i in range(len(approximate_solution))), 
            1.0,
            atol=1e-6
            ), "Approximate solution is not normalized."
        
        # calculate residual
        sol_norm = np.linalg.norm(approximate_solution)
        if sol_norm == 0:
            residual = np.nan
        else: 
            approximate_solution = approximate_solution / sol_norm
            # Scale the normalized solution to minimize the residual
            scaling_factor = self.norm_estimation(A, b, approximate_solution)
            scaled_solution = approximate_solution * scaling_factor
            residual = np.linalg.norm(b - A @ scaled_solution)

        return approximate_solution, success_rate, residual
    

    def process_swap_test(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
        swap_test_vector: np.ndarray
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

        total_shots = sum(result.values())
        
        for key, value in result.items():
            # postprocess shots to only consider those with ancilla measurement '1'
            if key[0] == '1': 
                correct_shots += value
                # among correct shots, count how many times the swap ancilla measurement is '1'
                if key[-1] == '1':
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
