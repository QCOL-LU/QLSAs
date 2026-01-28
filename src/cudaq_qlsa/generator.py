import numpy as np
from numpy.linalg import cond, eigvalsh, norm, solve


def _bandwidth_from_sparsity(n: int, sparsity: float) -> int:
    """Map a target sparsity (fraction of zeros) to a half-bandwidth."""
    if n <= 1:
        return 0

    density = 1 - sparsity
    density = min(max(density, 1.0 / n), 1.0)  # keep within sensible range

    approx_nonzeros_per_row = density * n
    bandwidth = int(np.ceil(max(approx_nonzeros_per_row - 1, 0) / 2))
    return min(max(bandwidth, 0), n - 1)


def _build_random_banded_matrix(n: int, bandwidth: int, rng: np.random.Generator) -> np.ndarray:
    """Create a symmetric banded matrix with random entries."""
    A = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        # Diagonal entries get moderate positive values
        A[i, i] = rng.uniform(0.5, 1.5)

        upper_start = i + 1
        upper_end = min(n, i + bandwidth + 1)
        for j in range(upper_start, upper_end):
            value = rng.normal(loc=0.0, scale=1.0 / max(bandwidth + 1, 1))
            A[i, j] = value
            A[j, i] = value

    return A


def generate_problem(
    n: int,
    cond_number: float = 5.0,
    sparsity: float = 0.5,
    seed: int | None = None,
    bandwidth: int | None = None,
    epsilon: float = 1e-6,
    return_permutation: bool = False,
):
    """
    Generate an Ax=b problem instance tailored for HHL benchmarking.

    This version constructs matrices by following four explicit steps:
    (1) build a random banded symmetric matrix with controllable bandwidth,
    (2) shift it to be positive definite,
    (3) reshape the eigenvalues to hit the requested condition number, and
    (4) apply a symmetric permutation to scramble sparsity structure.

    Parameters
    ----------
    n : int
        Matrix dimension (must be a positive power of two for HHL).
    cond_number : float, optional
        Target condition number (>= 1).
    sparsity : float, optional
        Target fraction of zero entries (0 <= sparsity < 1).
    seed : int, optional
        Seed for the random number generator.
    bandwidth : int, optional
        Half-bandwidth for the initial banded matrix. If None, it is inferred
        from the target sparsity.
    epsilon : float, optional
        Minimal eigenvalue buffer to enforce positive definiteness.
    return_permutation : bool, optional
        If True, include the permutation array used in step (4) in the result.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'A': generated Hermitian positive definite matrix.
        - 'b': right-hand side vector.
        - 'csol': exact solution of the linear system.
        - 'condition_number': measured 2-norm condition number of A.
        - 'sparsity': actual fraction of zero entries in A.
        - 'eigs': sorted eigenvalues of A (ascending).
        - 'bandwidth': half-bandwidth used to build the matrix.
        - 'permutation': permutation applied (only when return_permutation=True).
    """

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Problem size n={n} must be a positive power of 2 for HHL")

    if cond_number < 1:
        raise ValueError("cond_number must be >= 1")

    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in the interval [0, 1)")

    rng = np.random.default_rng(seed)

    if bandwidth is None:
        bandwidth = _bandwidth_from_sparsity(n, sparsity)
    else:
        bandwidth = int(bandwidth)
        if bandwidth < 0 or bandwidth >= n:
            raise ValueError(f"bandwidth must satisfy 0 <= bandwidth < n (got {bandwidth})")

    # Step 1: random symmetric banded matrix
    A = _build_random_banded_matrix(n, bandwidth, rng)

    # Step 2: enforce positive definiteness via diagonal shift
    evals_step1 = eigvalsh(A)
    lambda_min = evals_step1.min()
    if lambda_min <= 0:
        shift = -lambda_min + epsilon
    else:
        shift = epsilon
    A += shift * np.eye(n)

    # Step 3: eigen-transform to match desired condition number
    evals, vecs = np.linalg.eigh(A)
    lambda_min_current = evals.min()
    lambda_max_current = evals.max()

    if cond_number == 1:
        target_evals = np.ones_like(evals)
    elif np.isclose(lambda_max_current, lambda_min_current):
        target_evals = np.linspace(1.0, cond_number, num=n)
    else:
        scaled = (evals - lambda_min_current) / (lambda_max_current - lambda_min_current)
        target_evals = 1.0 + scaled * (cond_number - 1.0)

    # Reconstruct matrix with new eigenvalues
    A = (vecs * target_evals) @ vecs.T
    A = (A + A.T) / 2  # numerical symmetrization

    # Step 4: apply a symmetric permutation
    permutation = rng.permutation(n)
    A = A[np.ix_(permutation, permutation)]

    # Prepare right-hand side and normalize system
    b = rng.standard_normal(n)
    b = b[permutation]

    beta = max(norm(A), norm(b))
    if beta == 0:
        beta = 1.0
    A /= beta
    b /= beta

    csol = solve(A, b)

    result = {
        "A": A,
        "b": b,
        "csol": csol,
        "condition_number": cond(A),
        "sparsity": 1 - (np.count_nonzero(A) / float(A.size)),
        "eigs": eigvalsh(A),
        "bandwidth": bandwidth,
    }

    if return_permutation:
        result["permutation"] = permutation

    return result

