import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)
    if x.shape!=p.shape:
        raise ValueError("x and p shapes doesn't match")
    if not np.allclose(np.sum(p),1,atol=1e-6):
        raise ValueError("Probability don't sum to 1")
    return float(np.sum(x*p))
  
