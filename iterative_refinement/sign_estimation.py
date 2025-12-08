"""
Sign estimation for iterative refinement.
"""
import numpy as np
import itertools

def sign_estimation(A, b, x):
    """
    Sign estimation for iterative refinement. Finds solution z such that alpha = argmin ||A z - b||_2^2 and |z|=|x|.
    """
    n = len(x)
    matrix = []
    for bits in itertools.product([-1, 1], repeat=n):
        if any(column == list(bits) for column in matrix):
            continue
        matrix.append(list(bits))
    z = np.zeros(n)
    mins = np.infty
    for i in range(len(matrix)):
        t = np.linalg.norm(A@np.multiply(matrix[i], x) - b)
        if t < mins:
            mins = t
            z = np.multiply(matrix[i], x)
    return z