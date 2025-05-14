
import numpy as np
from numpy import linalg as LA
from numpy.linalg import cond, solve, eigvals

def generate_problem(n, cond_number=5, sparsity=0.5, is_diagonal=False, is_positive =False, integer_elements=False):
    """
    Generate a Hermitian matrix A of size n x n with a specified approximate condition number and sparsity.
    Then generate a random vector b, normalize them, and compute the solution csol = A^{-1} b.

    Parameters
    ----------
    n : int
        Size of the matrix.
    cond_number : float
        Desired condition number. The matrix is constructed by choosing eigenvalues between [1, cond_number].
    sparsity : float
        Desired fraction of off-diagonal elements to keep. Must be between 0 and 1.
        For example, 0.5 means keep 50% of off-diagonal elements at random, zeroing out the others.
    is_diagonal : bool, optional
        If True, generates a diagonal matrix. Defaults to False.
    is_positive : bool, optional
       Returns positive solution and positive b.
    integer_elements : bool, optional
        If True, generates a matrix with integer elements. Defaults to False.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'A': the generated Hermitian matrix
        - 'b': the right-hand side vector
        - 'csol': the exact solution A^{-1} b
        - 'condition_number': the measured condition number of A
        - 'sparsity': the actual sparsity level (fraction of zero elements)
        - 'eigs': sorted eigenvalues of A
    """
    from numpy.linalg import solve, norm, cond, eigvals
    from scipy.linalg import lstsq
    import numpy as np
    from scipy.optimize import minimize

    # Generate a random vector b
    b = np.random.randn(n).round(2)

    # Step 1: Construct desired eigenvalues
    eigenvalues = np.linspace(1, cond_number, n)

    # Step 2: Generate the matrix A
    if is_diagonal:
        A = np.diag(eigenvalues)  # Create a diagonal matrix
    else:
        # Generate a Hermitian matrix
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        Lambda = np.diag(eigenvalues)
        A = Q @ Lambda @ Q.conj().T

    # Step 3: Introduce sparsity (if not diagonal)
    if not is_diagonal:
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, True)
        off_diag_indices = np.where(~np.eye(n, dtype=bool))
        num_off_diag = len(off_diag_indices[0])
        num_keep = int(sparsity * num_off_diag)
        keep_indices = np.random.choice(num_off_diag, size=num_keep, replace=False)
        off_diag_mask = np.zeros(num_off_diag, dtype=bool)
        off_diag_mask[keep_indices] = True
        mask[off_diag_indices] = off_diag_mask
        for i in range(n):
            for j in range(i + 1, n):
                val = mask[i, j] and mask[j, i]
                mask[i, j] = val
                mask[j, i] = val
        A = A * mask

    if is_positive:
      csol = np.random.rand(n) + 0.1  # solution
      b = A @ csol 

    # Convert to integers if requested
    if integer_elements:
        A = A.astype(int)
        b = b.astype(int)

    # Normalize A and b, avoid division by zero
    beta = max(norm(A), norm(b))
    # If beta is zero regenerate 
    if beta == 0:
        return generate_problem(n, cond_number, sparsity, is_diagonal, negativity, integer_elements)

    A = A / beta
    b = b / beta

    # Check if A contains inf or NaN after normalization
    if not np.isfinite(A).all():
        return generate_problem(n, cond_number, sparsity, is_diagonal, negativity, integer_elements)

    # Compute exact solution, handling singular matrices
    try:
        csol = solve(A, b) #If A is singular, regenerate the problem
    except np.linalg.LinAlgError:
        return generate_problem(n, cond_number, sparsity, is_diagonal, negativity, integer_elements) #recursion call. if singular matrix encountered, regenerate a new problem.

    problem = {
        'A': A,
        'b': b,
        'csol': csol,
        'condition_number': cond(A),
        'sparsity': 1 - (np.count_nonzero(A) / float(A.size)),
        'eigs': np.sort(eigvals(A))
    }
    return problem






