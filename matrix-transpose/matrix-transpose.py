import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    rows = len(A)
    cols = len(A[0])
    res = np.zeros((cols,rows))
    for row in range(rows):
        for col in range(cols):
            res[col][row] = A[row][col]

    return res
    pass
