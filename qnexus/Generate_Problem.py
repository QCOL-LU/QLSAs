import numpy as np
from numpy import linalg as LA
from numpy.linalg import cond, solve, eigvals

def generate_problem(n, cond_number=5, sparsity=0.5, seed=None):
    # ... (rest of the file is identical to the original)
    if seed is not None:
        np.random.seed(seed)
    b = np.random.randn(n).round(2)
    eigenvalues = np.linspace(1, cond_number, n)
    X = np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    Lambda = np.diag(eigenvalues)
    A = Q @ Lambda @ Q.conj().T
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
    beta = max(LA.norm(A), LA.norm(b))
    if beta == 0:
        return generate_problem(n, cond_number, sparsity, seed) # Recurse if norms are zero
    A = A / beta
    b = b / beta
    A = A.round(2)
    b = b.round(2)
    
    try:
        csol = solve(A, b)
    except np.linalg.LinAlgError:
        print("Singular matrix encountered. Regenerating problem.")
        return generate_problem(n, cond_number, sparsity, seed+1 if seed else None)
        
    problem = {
        'A': A,
        'b': b,
        'csol': csol,
        'condition_number': cond(A),
        'sparsity': 1 - (np.count_nonzero(A) / float(A.size)),
        'eigs': np.sort(eigvals(A))
    }
    return problem