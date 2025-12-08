"""
Norm estimation for iterative refinement.
"""
import numpy as np

def norm_estimation(
    A: np.ndarray, 
    b: np.ndarray, 
    x: np.ndarray
) -> float:
    """
    Estimates the norm of the solution x to the linear system Ax = b.
    """
    v = A @ x
    denominator = np.dot(v, v)
    if denominator == 0:
        # If denominator is zero, the vector x is in the null space of A
        # This indicates a degenerate case. Return a small value to continue iteration.
        # In practice, this might indicate the system is ill-conditioned.
        return 1e-10  # Use a smaller value for better numerical stability
    return np.dot(v, b) / denominator