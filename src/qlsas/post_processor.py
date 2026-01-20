import numpy as np
import math
import numpy.linalg as LA
from qiskit.primitives.containers import SamplerPubResult

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
            counts: The counts dictionary consisting of measured bitstrings and their measurement counts.
        Returns:
            The normalized classical solution vector.
        """
        if isinstance(result, SamplerPubResult):
            return self.process_qiskit(result, A, b)
        else:
            raise ValueError(f"Invalid result type: {type(result)}.  Quantinuum result not yet supported.")

    def process_qiskit(
        self,
        result: SamplerPubResult,
        A: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:

        counts = result.join_data(names=['ancilla_flag_result', 'x_result']).get_counts()

        # extract amplitudes of x register
        x_size = int(math.log2(len(b)))
        num_successful_shots = 0 # for normalization
        approximate_solution = np.zeros(len(b))

        for key, value in counts.items():
            if key[-1] == '1': # ancilla measurement successful
                num_successful_shots += value
                coord = int(key[:x_size], base=2) # position in b vector from binary string
                approximate_solution[coord] = value
        if num_successful_shots == 0:
            raise ValueError("No successful shots.")
        else:
            approximate_solution = np.sqrt(approximate_solution/num_successful_shots)
        
        # extract signs of each solution coordinate, using classical solution for now (to be updated)
        classical_solution = LA.solve(A / LA.norm(b), b / LA.norm(b))
        for i in range(len(approximate_solution)):
            approximate_solution[i] = approximate_solution[i] * np.sign(classical_solution[i])
        
        assert np.allclose(
            sum(approximate_solution[i]**2 for i in range(len(approximate_solution))), 
            1.0,
            atol=1e-6
            ), "Approximate solution is not normalized."
        return approximate_solution