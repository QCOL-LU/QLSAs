from __future__ import annotations

from typing import Optional, Union

import numpy as np
from qiskit.providers.backend import BackendV1, BackendV2
from qnexus import QuantinuumConfig

from qlsas.executer import Executer
from qlsas.post_processor import Post_Processor
from qlsas.qlsa.base import QLSA
from qlsas.transpiler import Transpiler


class QuantumLinearSolver:
    """End-to-end runner for a QLSA: build -> transpile -> execute -> post-process."""

    def __init__(
        self,
        qlsa: QLSA,
        backend: Union[BackendV1, BackendV2, QuantinuumConfig],
        *,
        shots: int = 1024,
        optimization_level: int = 3,
        executer: Optional[Executer] = None,
        post_processor: Optional[Post_Processor] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.qlsa = qlsa
        self.backend = backend
        self.shots = shots
        self.optimization_level = optimization_level
        self.executer = executer or Executer()
        self.post_processor = post_processor or Post_Processor()
        self.mode = mode
        
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Run the full workflow and return the (post-processed) solution vector."""
        circuit = self.qlsa.build_circuit(A, b)

        transpiler = Transpiler(
            circuit=circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        transpiled_circuit = transpiler.optimize()

        result = self.executer.run(transpiled_circuit, self.backend, self.shots)
        return self.post_processor.process(result, A, b)

