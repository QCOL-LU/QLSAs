import cudaq
import numpy as np
import math
import numpy.linalg as LA


class Post_Processor:
    """
    Post_Processor class for post-processing the result of the quantum linear solver.
    """

    def process(
        self, 
        result,
        A: np.ndarray,
        b: np.ndarray,
        ):
        """
        Process the result of the quantum linear solver and return the solution vector.
        Args:
            items: The counts dictionary consisting of measured bitstrings and their measurement counts.
        Returns:
            The normalized classical solution vector.
        """
        return self.process_cudaq(result, A, b)


    def process_cudaq(
        self,
        result,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:

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
        
        # extract signs of each solution coordinate, using classical solution for now (to be updated)
        classical_solution = LA.solve(A / LA.norm(b), b / LA.norm(b))
        # for i in range(len(approximate_solution)):
        #     approximate_solution[i] = approximate_solution[i] * np.sign(classical_solution[i])
        for idx in range(len(classical_solution)):
            if np.sign(classical_solution[idx]) != np.sign(approximate_solution[idx]):
                approximate_solution[idx] = (-1)*approximate_solution[idx]
        
        assert np.allclose(
            sum(approximate_solution[i]**2 for i in range(len(approximate_solution))), 
            1.0,
            atol=1e-6
            ), "Approximate solution is not normalized."
        return approximate_solution