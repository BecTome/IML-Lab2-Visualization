import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import svd_flip


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components_ = n_components
        self.components_ = None

    def fit(self, X: np.ndarray) -> None:
        # Center data
        X -= np.mean(X, axis=0)

        # Calculate covariance matrix
        cov = np.cov(X, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n eigenvectors
        self.components_ = eigenvectors[:, :self.n_components_]
        

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        transformed_X = self.transform(X)

        return transformed_X

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Center data
        X -= np.mean(X, axis=0)

        # Project data onto eigenvectors
        transformed_X = np.dot(X, self.components_)

        return -transformed_X
