from __future__ import annotations

from typing import Optional

import numpy as np

from cudaq_qlsa.qlsa.base import QLSA
from cudaq_qlsa.executer import Executer
from cudaq_qlsa.post_processor import Post_Processor


class QuantumLinearSolver:
    """End-to-end runner for a QLSA: build -> transpile -> execute -> post-process."""

    def __init__(
        self,
        qlsa: QLSA,
        backend: str,
        *,
        shots: int = 1024,
        noisemodel: Optional[Any] = None,
        executer: Optional[Executer] = None,
        post_processor: Optional[Post_Processor] = None,
        verbose: bool = True
    ) -> None:
        self.qlsa = qlsa
        self.backend = backend
        self.shots = shots
        self.noise_model = noisemodel 
        self.executer = executer or Executer()
        self.post_processor = post_processor or Post_Processor()
        self.verbose = verbose

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Run the full workflow and return the (post-processed) solution vector."""
        kernel, args = self.qlsa.build_circuit(A, b)
    
        result = self.executer.run(kernel, args, self.backend, self.shots, self.noise_model, self.verbose)
        return self.post_processor.process(result, A, b)
