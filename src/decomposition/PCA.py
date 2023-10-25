import numpy as np
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components_ = n_components
        self.components_ = None

    def fit(self, X: np.ndarray) -> None:
        # Center data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Compute the covariance matrix of the data.
        covariance_matrix = np.cov(X)

        # Get the eigenvectors and eigenvalues to determine the principal
        # components of the data.
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvectors in decreasing order of eigenvalues (from bigger to smaller).
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        # Select the desired first n components
        self.components_ = eigenvectors[self.n_components_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        transformed_X = self.transform(X)

        return transformed_X

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Standarize the data before transforming it.
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        transformed_X = self.components_.T * X.T

        return transformed_X
