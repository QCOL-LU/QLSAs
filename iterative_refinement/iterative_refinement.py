"""
Iterative refinement utilities for quantum linear system solvers.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, solve

from norm_estimation import norm_estimation


class IterativeRefinement:
    """
    Driver for iterative refinement that is agnostic to the underlying solver.

    The solver is expected to expose a ``solve`` method (e.g. a LinearSystemSolver)
    that returns an object with either a ``solution``/``vector`` attribute or a
    dictionary containing the solved vector under the ``x`` or ``solution`` key.
    """

    def __init__(
        self,
        solver: Any,
        *,
        solver_options: Optional[Dict[str, Any]] = None,
        backend_label: Optional[str] = None,
    ) -> None:
        self.solver = solver
        self.default_solver_options: Dict[str, Any] = solver_options or {}
        self.backend_label = backend_label or self._infer_backend_label()

    def run(
        self,
        A: np.ndarray,
        b: np.ndarray,
        *,
        precision: float,
        max_iter: int,
        solver_options: Optional[Dict[str, Any]] = None,
        plot: bool = True,
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform iterative refinement around the provided solver.

        Args:
            A: Coefficient matrix.
            b: Right-hand side vector.
            precision: Desired residual norm to stop iterations.
            max_iter: Maximum number of refinement iterations.
            solver_options: Extra keyword arguments forwarded to the solver.
            plot: If True, saves residual/error plots.
            data_dir: Optional override for where plots are written.
        """

        nabla, rho, d = 1.0, 2.0, len(A)
        iteration = 0
        x = np.zeros(d)
        csol = solve(A, b)
        csol_normalized = csol / norm(csol)
        res_list: list[float] = []
        error_list: list[float] = []

        print("IR: Obtaining initial solution...")
        base_solver_kwargs = {**self.default_solver_options, **(solver_options or {})}
        initial_solution = self._solve(A, b, {**base_solver_kwargs, "iteration": 0})
        x = self._extract_solution(initial_solution)

        r = b - A @ x
        x_normalized = self._normalize(x, "initial solver output")
        error_list.append(norm(csol_normalized - x_normalized))
        res_list.append(norm(r))
        print(f"Initial residual: {res_list[0]:.4f}, Initial error: {error_list[0]:.4f}\n")

        iteration = 1
        while norm(r) > precision and iteration <= max_iter:
            print(f"IR Iteration: {iteration}")
            new_r = nabla * r
            solver_result = self._solve(
                A, new_r, {**base_solver_kwargs, "iteration": iteration}
            )
            x_new = self._extract_solution(solver_result)
            alpha = norm_estimation(A, new_r, x_new)
            x += (alpha / nabla) * x_new
            x_normalized = self._normalize(x, "refined solution")

            r = b - A @ x
            err = norm(csol_normalized - x_normalized)
            res = norm(r)
            error_list.append(err)
            res_list.append(res)

            print(f"  residual: {res:.4f}, error: {err:.4f}, alpha: {alpha:.4f}\n")

            if res < 1e-9:
                nabla *= rho
            else:
                nabla = min(rho * nabla, 1 / res)
            iteration += 1

        result: Dict[str, Any] = {
            "refined_x": x,
            "residuals": res_list,
            "errors": error_list,
            "total_iterations": iteration - 1,
            "initial_solution": initial_solution,
            "backend_label": self.backend_label,
        }

        if plot:
            result["plot_paths"] = self._plot_results(
                res_list,
                error_list,
                b,
                base_solver_kwargs.get("n_qpe_qubits"),
                data_dir=data_dir,
            )

        return result

    def _solve(self, A: np.ndarray, b: np.ndarray, options: Dict[str, Any]) -> Any:
        """Invoke the provided solver with merged options, tolerating missing iteration kwargs."""

        solve_fn = getattr(self.solver, "solve", None)
        if not callable(solve_fn):
            raise TypeError("Provided solver must implement a 'solve' method.")

        try:
            return solve_fn(A, b, **options)
        except TypeError as exc:
            # Allow solvers that do not accept the synthetic 'iteration' kwarg.
            if "iteration" in options:
                fallback = {k: v for k, v in options.items() if k != "iteration"}
                try:
                    return solve_fn(A, b, **fallback)
                except TypeError:
                    pass
            raise exc

    @staticmethod
    def _extract_solution(result: Any) -> np.ndarray:
        """
        Extract a solution vector from different solver return types.

        Supports SolverResult-like objects, dictionaries, or raw numpy arrays.
        """

        if isinstance(result, np.ndarray):
            return result
        if isinstance(result, dict):
            if "x" in result:
                return result["x"]
            if "solution" in result:
                return result["solution"]
        if hasattr(result, "solution"):
            return result.solution  # type: ignore[attr-defined]
        if hasattr(result, "vector"):
            return result.vector  # type: ignore[attr-defined]
        raise TypeError("Solver result does not contain a solution vector.")

    @staticmethod
    def _normalize(vector: np.ndarray, context: str) -> np.ndarray:
        """Normalize a vector and guard against degenerate outputs."""

        vec_norm = norm(vector)
        if np.isclose(vec_norm, 0.0):
            raise ValueError(f"{context} returned a zero vector; cannot normalise.")
        return vector / vec_norm

    def _infer_backend_label(self) -> str:
        """Best-effort label for plots without assuming solver internals."""

        candidate_attrs = ("backend_label", "backend_name", "backend")
        for attr in candidate_attrs:
            value = getattr(self.solver, attr, None)
            if isinstance(value, str):
                return value
            if hasattr(value, "name"):
                return value.name  # type: ignore[return-value]
        return self.solver.__class__.__name__

    def _plot_results(
        self,
        residuals: list[float],
        errors: list[float],
        b: np.ndarray,
        n_qpe_qubits: Optional[int],
        *,
        data_dir: Optional[str],
    ) -> Dict[str, str]:
        """Create and save residual and error plots."""

        iterations_range = np.arange(len(residuals))
        size = len(b)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved_data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(resolved_data_dir, exist_ok=True)

        n_qpe_label = (
            f"_qpe{n_qpe_qubits}" if n_qpe_qubits is not None else "_qpe-"
        )
        backend_label = self.backend_label

        plt.figure()
        plt.plot(
            iterations_range,
            [np.log10(r) for r in residuals],
            "o--",
            label=f"{size}x{size} on {backend_label}",
        )
        plt.ylabel(r"$\log_{10}(\|Ax-b\|_2)$")
        plt.xlabel("IR Iteration")
        plt.legend()
        plt.title("Residual Norm vs. Iteration")
        plt.tight_layout()
        residuals_filename = (
            f"plot_residuals_{backend_label}_{size}x{size}{n_qpe_label}_{timestamp}.png"
        )
        residuals_path = os.path.join(resolved_data_dir, residuals_filename)
        plt.savefig(residuals_path)

        plt.figure()
        plt.plot(
            iterations_range,
            [np.log10(e) for e in errors],
            "o--",
            label=f"{size}x{size} on {backend_label}",
        )
        plt.ylabel(r"$\log_{10}(\|x-x_{\mathrm{classical}}\|_2)$")
        plt.xlabel("IR Iteration")
        plt.legend()
        plt.title("Solution Error vs. Iteration")
        plt.tight_layout()
        errors_filename = (
            f"plot_errors_{backend_label}_{size}x{size}{n_qpe_label}_{timestamp}.png"
        )
        errors_path = os.path.join(resolved_data_dir, errors_filename)
        plt.savefig(errors_path)

        return {"residuals_plot": residuals_path, "errors_plot": errors_path}


def IR(
    A: np.ndarray,
    b: np.ndarray,
    precision: float,
    max_iter: int,
    solver: Any,
    *,
    solver_options: Optional[Dict[str, Any]] = None,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Backwards-compatible helper to run iterative refinement without manually
    instantiating the class. Prefer creating an ``IterativeRefinement`` object.
    """

    refinement = IterativeRefinement(solver, solver_options=solver_options)
    return refinement.run(
        A,
        b,
        precision=precision,
        max_iter=max_iter,
        solver_options=solver_options,
        plot=plot,
    )
