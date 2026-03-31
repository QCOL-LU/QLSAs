from __future__ import annotations

from collections import defaultdict
from typing import Optional, Union

import numpy as np
from qiskit.providers.backend import BackendV2

from qlsas.quantinuum_config import QuantinuumBackendConfig
from qlsas.executer import Executer
from qlsas.ibm_options import IBMExecutionOptions
from qlsas.post_processor import Post_Processor
from qlsas.algorithms.base import QLSA
from qlsas.transpiler import Transpiler


class QuantumLinearSolver:
    """End-to-end runner for a QLSA: build -> transpile -> execute -> post-process."""

    def __init__(
        self,
        qlsa: QLSA,
        backend: Union[BackendV2, QuantinuumBackendConfig],
        *,
        shots: int = 1024,
        target_successful_shots: Optional[int] = None,
        shots_per_batch: Optional[int] = None,
        max_total_shots: Optional[int] = None,
        optimization_level: int = 3,
        ibm_options: Optional[IBMExecutionOptions] = None,
        executer: Optional[Executer] = None,
        post_processor: Optional[Post_Processor] = None,
        
    ) -> None:
        self.qlsa = qlsa
        self.backend = backend
        self.shots = shots
        self.target_successful_shots = target_successful_shots
        self.shots_per_batch = shots_per_batch or shots
        self.max_total_shots = max_total_shots
        self.optimization_level = optimization_level
        self.ibm_options = ibm_options
        self.executer = executer or Executer(ibm_options=ibm_options)
        self.post_processor = post_processor or Post_Processor()


    @property
    def _is_quantinuum(self) -> bool:
        return isinstance(self.backend, QuantinuumBackendConfig)

    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        t0: Optional[float] = None,
        C: Optional[float] = None,
    ) -> np.ndarray:
        """Run the full workflow and return the (post-processed) solution vector."""
        self.circuit = self.qlsa.build_circuit(A, b, t0=t0, C=C)

        transpiler = Transpiler(
            circuit=self.circuit,
            backend=self.backend,
            optimization_level=self.optimization_level,
        )
        self.transpiled_circuit = transpiler.optimize()

        if self.qlsa.readout == "swap_test":
            result = self._execute(transpiler, verbose=verbose)
            return self.post_processor.process_swap_test(
                result, A, b, self.qlsa.swap_test_vector
            )[0]

        if self.qlsa.readout != "measure_x":
            raise ValueError(
                f"Invalid readout method: {self.qlsa.readout}. "
                "Must be 'measure_x' or 'swap_test'."
            )

        if self.target_successful_shots is not None:
            return self._solve_until_successful_shots(
                self.transpiled_circuit, A, b, verbose=verbose,
                transpiler=transpiler,
            )

        result = self._execute(transpiler, verbose=verbose)
        return self.post_processor.process_tomography(result, A, b, verbose=verbose)[0]

    def _execute(
        self, transpiler: Transpiler, verbose: bool = True, shots: int | None = None,
    ):
        """Dispatch execution to the correct backend path."""
        effective_shots = shots if shots is not None else self.shots
        if self._is_quantinuum:
            return self.executer.run(
                self.transpiled_circuit,
                self.backend,
                effective_shots,
                verbose=verbose,
                register_infos=transpiler.register_infos,
                measurement_plan=transpiler.measurement_plan,
                optimization_level=self.optimization_level,
            )
        return self.executer.run(
            self.transpiled_circuit,
            self.backend,
            effective_shots,
            ibm_options=self.ibm_options,
            verbose=verbose,
        )

    def _solve_until_successful_shots(
        self,
        transpiled_circuit,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        transpiler: Transpiler | None = None,
    ) -> np.ndarray:
        """Run batches of shots until we have at least target_successful_shots with ancilla=1.

        For Quantinuum backends, all shots are run in a single execution (no
        per-batch Guppy recompilation) and the result is trimmed client-side.
        For IBM backends, shots are submitted in batches within an IBM session.
        """
        if self._is_quantinuum:
            return self._quantinuum_successful_shots(
                A, b, verbose=verbose, transpiler=transpiler,
            )
        return self._ibm_successful_shots(
            transpiled_circuit, A, b, verbose=verbose,
        )

    def _quantinuum_successful_shots(
        self,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
        transpiler: Transpiler | None = None,
    ) -> np.ndarray:
        """Single-execution path: run max_total_shots (or a large batch) once, trim to target."""
        assert transpiler is not None
        total_shots = self.max_total_shots or self.shots
        counts = self._execute(transpiler, verbose=verbose, shots=total_shots)
        trimmed = _trim_counts_to_target(counts, self.target_successful_shots)
        return self.post_processor.tomography_from_counts(trimmed, A, b)[0]

    def _ibm_successful_shots(
        self,
        transpiled_circuit,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """Batched IBM path: submit batches within a session until target is reached."""
        accumulated: dict[str, int] = defaultdict(int)
        num_successful_so_far = 0
        total_shots_so_far = 0
        batch_size = self.shots_per_batch

        opened_session = False
        if self.executer.session_active:
            self.executer.open_session(self.backend, verbose=verbose)
            opened_session = self.executer.session_active

        try:
            while True:
                result = self.executer.run(
                    transpiled_circuit,
                    self.backend,
                    batch_size,
                    ibm_options=self.ibm_options,
                    verbose=verbose,
                )
                joined = result.join_data(names=["ancilla_flag_result", "x_result"])
                counts = joined.get_counts()

                num_batch_successful = sum(v for k, v in counts.items() if k[-1] == "1")

                if (
                    self.target_successful_shots is not None
                    and num_successful_so_far + num_batch_successful
                    <= self.target_successful_shots
                ):
                    for key, value in counts.items():
                        accumulated[key] += value
                    num_successful_so_far += num_batch_successful
                    total_shots_so_far += sum(counts.values())

                    if num_successful_so_far == self.target_successful_shots:
                        break
                else:
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
        finally:
            if opened_session:
                self.executer.close_session(verbose=verbose)

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


def _trim_counts_to_target(
    counts: dict[str, int],
    target_successful: int | None,
) -> dict[str, int]:
    """Keep all failed shots and exactly *target_successful* successful shots (ancilla=1)."""
    if target_successful is None:
        return counts

    trimmed: dict[str, int] = {}
    found = 0
    for key, count in counts.items():
        if key[-1] != "1":
            trimmed[key] = count
            continue
        if found >= target_successful:
            continue
        take = min(count, target_successful - found)
        trimmed[key] = take
        found += take

    return trimmed
