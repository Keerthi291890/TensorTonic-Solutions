import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    #centering the data
    X = np.array(X)
    mean = np.mean(X,axis=0)
    X_c = X-mean
    n = len(X[0])
    #Computing the covariance matrix
    C = (X_c.T@X_c)/(n-1)
    #Find out eigenvectors and eigenvalues of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    W = eigenvectors[:,0:k]
    X_proj = X_c@W
    return X_proj
    
    