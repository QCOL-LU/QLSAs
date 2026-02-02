import cudaq
import warnings

from typing import Any, Optional

class Executer:
    """
    Executer class for running circuits on target backends.
    Must specify a hardware or simulator backend from cudaq.
    """

    def run(
        self, 
        kernel, 
        args,
        backend: str,
        shots: int,
        noise_model: Optional[Any] = None,
        verbose: bool = True
    ):
        """
        Run the circuit on the backend.
        Args:
            circuit: The circuit to execute.
            backend: The backend to execute the circuit on.
            shots: The number of shots to run.
        Returns:
            A result object containing the result of the execution.
        """
        # Setting the target backend
        if backend != cudaq.get_target().name and cudaq.has_target(backend):
            cudaq.set_target(backend)
            print(f'Target changed to {backend}.')
        elif backend == cudaq.get_target().name:
            pass
        else:
            raise ValueError(f"Invalid backend: {backend}. Must be 'nvidia', 'qpp-cpu', or 'density-matrix-cpu'")

        # Configuring noise in the simulation
        if noise_model:
            if backend == 'qpp-cpu':
                warnings.warn("In order to run a noisy simulation on CPU, change the backend to density-matrix-cpu. The backend qpp-cpu returns a noiseless simulation.")
            else:
                if verbose:
                    print(f'Noisy simulation underway ...')
                return self.run_cudaq(kernel, args, shots, noise_model)
        else:
            if verbose:
                print(f'Noiseless simulation underway ...')
            return self.run_cudaq(kernel, args, shots)

    
    def run_cudaq(
        self, 
        kernel, 
        args,
        shots: int,
        noise_model: Optional[Any] = None
    ) -> dict:
        """
        Run the circuit on the cudaq backend.
        """
        result = cudaq.sample(kernel, *args, shots_count=shots, noise_model=noise_model)
        return result
    