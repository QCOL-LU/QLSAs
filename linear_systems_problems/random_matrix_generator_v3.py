import numpy as np
from numpy.linalg import cond, inv, norm, solve
from scipy.stats import unitary_group

#######################################################################################################################
#                                             Generation functions
#######################################################################################################################


def generate_random_sparse_matrix_sparsity(N, d):
    """
    Generate a random sparse matrix with a maximum sparsity.

    Parameters:
        N (int): Number of rows.
        m (int): Number of columns.
        max_sparsity (int): The maximum sparsity of the matrix (maximum non-zero entries in any row or column).

    Returns:
        k (float): condition number,
        x_norm (float): norm of the solution vector of the linear solver,
        max_norm (float): max norm of the generated matrix,
        one_norm (float): one norm of the generated matrix
    """
    if d < 0 or d > N:
        raise ValueError("Maximum sparsity must be between 0 and n")

    # Create an empty matrix filled with zeros.
    A = np.zeros((N, N), dtype=complex)

    # Fill the matrix with non-zero values based on the maximum sparsity except on the diagonal

    if d > 0:
        # Select a random row with max sparsity, i.e. d - 1
        random_row = np.random.randint(0, N)

        for i in range(N):
            # Randomly choose the number of non-zero elements in this row.
            num_nonzero = (
                np.random.randint(0, d) if i != np.random.randint(0, N) else d - 1
            )

            if num_nonzero > 0:
                # Randomly select columns for non-zero entries excluding the diagonal
                pool = np.delete(np.arange(N), i)
                nonzero_cols = np.random.choice(pool, num_nonzero, replace=False)

                # Assign random complex values to the selected columns.
                A[i, nonzero_cols] = np.random.uniform(
                    -1, 1, size=num_nonzero
                ) + 1j * np.random.uniform(-1, 1, size=num_nonzero)

    # make the matrix positive and invertible by multipling by the (random) indentity
    random_array = np.random.rand(N)
    max_A = np.max(A) if np.max(A) != 0 else 1
    A = A + max_A * np.diag(random_array)

    # make it Hermitian
    A = A + A.conj().T

    # Normalize
    A /= norm(A, ord=2)

    # Compute the condition number
    A_inv = inv(A)
    k = norm(A_inv, ord=2)

    # compute norms
    max_norm = np.max(np.abs(A))
    one_norm = norm(A, ord=1)

    # compute the norm of x
    b = np.random.uniform(-1, 1, N) + 1j * np.random.uniform(-1, 1, N)
    b /= norm(b)
    x_norm = norm(A_inv @ b)

    return k, x_norm, max_norm


def generate_random_diagonal_matrix_fixed_condition_number(N, k):
    """Generates a diagonal matrix of dimension NxN with condition number k"""

    random_eigenvalues = np.random.uniform(1 / k, 1, N - 2)

    # Include the eigenvalues 1 and 1/k
    eigenvalues = np.concatenate(([1], [1 / k], random_eigenvalues))

    # Shuffle the eigenvalues to ensure randomness in their positions
    np.random.shuffle(eigenvalues)

    # Build the matrix
    diagonal_matrix = np.diag(eigenvalues)

    return diagonal_matrix


def generate_problem(n, cond_number=5.0, sparsity=0.5, seed=None):
    """Generate an HHL problem instance with controllable condition and sparsity.

    Parameters
    ----------
    n : int
        Matrix size.
    cond_number : float, optional
        Requested condition number (>= 1).
    sparsity : float | int, optional
        Either:
        - float in (0, 1]: normalized block-size fraction (d = ceil(sparsity * n)),
        - int in [1, n]: maximum block size d directly.
    seed : int | None, optional
        Random seed.
    """
    if seed is not None:
        np.random.seed(seed)

    N = n
    k = cond_number
    if N <= 0:
        raise ValueError("n must be a positive integer")
    if k < 1:
        raise ValueError("cond_number must be >= 1")

    if isinstance(sparsity, (float, np.floating)):
        if not (0 < sparsity <= 1):
            raise ValueError("float sparsity must satisfy 0 < sparsity <= 1")
        d = int(np.ceil(float(sparsity) * N))
    else:
        d = int(sparsity)
    if d < 1 or d > N:
        raise ValueError("sparsity must satisfy 1 <= sparsity <= n")

    # Build block dimensions that sum to N, with each block size <= d.
    # Prefer blocks >= 2 when possible, while still allowing 1x1 blocks
    # for edge cases (e.g., d == 1 or odd leftovers).
    unitary_dim = []
    remaining = N
    while remaining > 0:
        if remaining <= d:
            unitary_dim.append(remaining)
            break

        max_next = min(d, remaining - 1)  # keep at least one element for later
        min_next = 2 if max_next >= 2 else 1
        next_value = np.random.randint(min_next, max_next + 1)
        unitary_dim.append(next_value)
        remaining -= next_value

    # Shuffle
    np.random.shuffle(unitary_dim)

    # Build the unitary
    U = np.zeros((N, N), dtype=complex)
    start_idx = 0

    # Generate each Ui in the corresponding block
    for i in range(len(unitary_dim)):
        Ui = (
            unitary_group.rvs(unitary_dim[i])
            if unitary_dim[i] > 1
            else np.exp(1j * np.random.uniform(0, 2 * np.pi))
        )
        U[
            start_idx : start_idx + unitary_dim[i],
            start_idx : start_idx + unitary_dim[i],
        ] = Ui
        start_idx += unitary_dim[i]

    D = generate_random_diagonal_matrix_fixed_condition_number(N, k)
    H = U @ D @ U.conj().T
    permutation = np.random.permutation(N)
    H = H[np.ix_(permutation, permutation)]

    # Compute solution vector
    b = np.random.uniform(-1, 1, N) + 1j * np.random.uniform(-1, 1, N)
    b /= norm(b)
    csol = solve(H, b)
    x_norm = norm(csol)

    # Compute relevant quantities for downstream analysis
    max_norm = np.max(np.abs(H))
    actual_sparsity = 1 - (np.count_nonzero(H) / float(H.size))

    return {
        "A": H,
        "b": b,
        "csol": csol,
        "condition_number": cond(H),
        "sparsity": actual_sparsity,
        "x_norm": x_norm,
        "max_norm": max_norm,
        "block_size": d,
        "permutation": permutation,
    }