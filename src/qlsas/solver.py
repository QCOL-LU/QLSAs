from __future__ import annotations

from collections import defaultdict
from typing import Optional, Union

import numpy as np
from qiskit.providers.backend import BackendV2
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
        backend: Union[BackendV2, QuantinuumConfig],
        *,
        shots: int = 1024,
        target_successful_shots: Optional[int] = None,
        shots_per_batch: Optional[int] = None,
        max_total_shots: Optional[int] = None,
        optimization_level: int = 3,
        executer: Optional[Executer] = None,
        post_processor: Optional[Post_Processor] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.qlsa = qlsa
        self.backend = backend
        self.shots = shots
        self.target_successful_shots = target_successful_shots
        self.shots_per_batch = shots_per_batch or shots
        self.max_total_shots = max_total_shots
        self.optimization_level = optimization_level
        self.executer = executer or Executer()
        self.post_processor = post_processor or Post_Processor()
        self.mode = mode

    def solve(self, A: np.ndarray, b: np.ndarray, verbose: bool = True, t0: Optional[float] = None, C: Optional[float] = None) -> np.ndarray:
        """Run the full workflow and return the (post-processed) solution vector."""
        self.circuit = self.qlsa.build_circuit(A, b, t0=t0, C=C)

        transpiler = Transpiler(
            circuit=self.circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        self.transpiled_circuit = transpiler.optimize()

        if self.qlsa.readout == "swap_test":
            result = self.executer.run(self.transpiled_circuit, self.backend, self.shots, mode=self.mode, verbose=verbose)
            return self.post_processor.process_swap_test(
                result, A, b, self.qlsa.swap_test_vector
            )[0]

        if self.qlsa.readout != "measure_x":
            raise ValueError(
                f"Invalid readout method: {self.qlsa.readout}.  Must be 'measure_x' or 'swap_test'."
            )

        if self.target_successful_shots is not None:
            return self._solve_until_successful_shots(self.transpiled_circuit, A, b, verbose=verbose)

        result = self.executer.run(self.transpiled_circuit, self.backend, self.shots, mode=self.mode, verbose=verbose)
        return self.post_processor.process_tomography(result, A, b, verbose=verbose)[0]

    def _solve_until_successful_shots(
        self, transpiled_circuit, A: np.ndarray, b: np.ndarray, verbose: bool = True
    ) -> np.ndarray:
        """Run batches of shots until we have at least target_successful_shots with ancilla=1."""
        accumulated: dict[str, int] = defaultdict(int)
        num_successful_so_far = 0
        total_shots_so_far = 0
        batch_size = self.shots_per_batch

        while True:
            result = self.executer.run(transpiled_circuit, self.backend, batch_size, mode=self.mode, verbose=verbose)
            joined = result.join_data(names=["ancilla_flag_result", "x_result"])
            counts = joined.get_counts()

            num_batch_successful = sum(v for k, v in counts.items() if k[-1] == "1")

            if (
                self.target_successful_shots is not None
                and num_successful_so_far + num_batch_successful
                <= self.target_successful_shots
            ):
                # Take the whole batch
                for key, value in counts.items():
                    accumulated[key] += value
                num_successful_so_far += num_batch_successful
                total_shots_so_far += sum(counts.values())

                if num_successful_so_far == self.target_successful_shots:
                    break
            else:
                # Take partial batch to hit target exactly
                needed = (
                    self.target_successful_shots - num_successful_so_far
                    if self.target_successful_shots is not None
                    else 0
                )
                bitstrings = joined.get_bitstrings()

                count_found = 0
                cutoff = 0
                for i, bs in enumerate(bitstrings):
                    if bs[-1] == "1":
                        count_found += 1
                    if count_found == needed:
                        cutoff = i + 1
                        break

                for bs in bitstrings[:cutoff]:
                    accumulated[bs] += 1

                break

            if (
                self.max_total_shots is not None
                and total_shots_so_far >= self.max_total_shots
            ):
                break

        final_successful = sum(v for k, v in accumulated.items() if k[-1] == "1")
        hit_max_limit = (
            self.max_total_shots is not None
            and total_shots_so_far >= self.max_total_shots
        )
        if self.target_successful_shots is not None and not hit_max_limit:
            assert final_successful == self.target_successful_shots, (
                f"Expected exactly {self.target_successful_shots} successful shots, "
                f"but got {final_successful}."
            )

        return self.post_processor.tomography_from_counts(dict(accumulated), A, b)[0]

