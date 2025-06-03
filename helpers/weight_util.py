import numpy as np

def initialize_weight(shape, method=None, seed=None):
    """
    Generates weight based on given method.

    Input:
        - shape: shape of matrix in the form: (N, D)
        - method: weight initialization method, options are "he" and "xavier"
    """
    assert method in ["he", "xavier"], "Method is not valid, possible options are: he, xavier"
    
    if seed is not None:
        np.random.seed(seed)

    N, D = shape
    if not method:
        return np.random.randn(*shape)

    if method == "xavier":
        return np.random.randn(*shape) * np.sqrt(1.0 / N)
    
    elif method == "he":
        return np.random.randn(*shape) * np.sqrt(1.0 / N+D)
