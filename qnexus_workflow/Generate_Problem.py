import numpy as np
from numpy import linalg as LA
from numpy.linalg import cond, solve, eigvals

# Generate Ax=b problem instance
def generate_problem(n, cond_number=5, sparsity=0.5, seed=None):
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
    seed : int, optional
        Random seed for reproducibility.

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
    if seed is not None:
        np.random.seed(seed)
    
    # Validate that n is a power of 2 (required for HHL algorithm)
    if not (n > 0 and (n & (n - 1)) == 0):
        raise ValueError(f"Problem size n={n} must be a power of 2 for the HHL algorithm")
    
    # Generate a random vector b
    b = np.random.randn(n).round(2)
    # Step 1: Construct desired eigenvalues to achieve the target condition number.
    # We set eigenvalues linearly spaced from 1 to cond_number.
    # This ensures before sparsity manipulation, cond(A) ~ cond_number.
    eigenvalues = np.linspace(1, cond_number, n)
    # Step 2: Generate a random unitary matrix Q via QR decomposition.
    # Start with a random complex or real matrix and orthonormalize.
    X = np.random.randn(n, n)  # + 1j*np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    # Step 3: Construct A = Q * diag(eigenvalues) * Q^H
    Lambda = np.diag(eigenvalues)
    A = Q @ Lambda @ Q.conj().T  # Hermitian by construction
    # Step 4: Introduce sparsity. We want a fraction 'sparsity' of the off-diagonal elements to remain.
    # The fraction of zeroed elements will be (1 - sparsity).
    # We'll create a mask for the off-diagonal elements and apply it symmetrically.
    mask = np.ones((n, n), dtype=bool)
    # We don't touch the diagonal.
    np.fill_diagonal(mask, True)
    # Create a random mask for off-diagonal entries
    off_diag_indices = np.where(~np.eye(n, dtype=bool))
    num_off_diag = len(off_diag_indices[0])
    # Number of off-diagonal elements to keep
    num_keep = int(sparsity * num_off_diag)
    # Choose which off-diagonal elements to keep
    keep_indices = np.random.choice(num_off_diag, size=num_keep, replace=False)
    # Create a boolean array for these positions
    off_diag_mask = np.zeros(num_off_diag, dtype=bool)
    off_diag_mask[keep_indices] = True
    # Apply to full matrix
    mask[off_diag_indices] = off_diag_mask
    # Ensure symmetry: mask[i,j] = mask[j,i], so apply symmetrical logic
    # If we zero out (i,j), also zero out (j,i).
    # Actually, since we picked symmetrical sets from scratch, just ensure symmetry now:
    for i in range(n):
        for j in range(i+1, n):
            # If one side is False, set both sides False
            # If one side is True, set both sides True
            val = mask[i, j] and mask[j, i]
            mask[i, j] = val
            mask[j, i] = val
    # Apply mask to A
    A = A * mask
    # A is still Hermitian because we applied the mask symmetrically.
    # However, the condition number might have changed.
    # Normalize A and b
    beta = max(LA.norm(A), LA.norm(b))
    A = A / beta
    b = b / beta
    # Round to 2 decimal places
    A = A.round(2)
    b = b.round(2)
    # Compute exact solution
    csol = solve(A, b)
    problem = {
        'A': A,
        'b': b,
        'csol': csol,
        'condition_number': cond(A),
        'sparsity': 1 - (np.count_nonzero(A) / float(A.size)),
        'eigs': np.sort(eigvals(A))
    }
    return problem