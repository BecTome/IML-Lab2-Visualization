import numpy as np
import pandas as pd
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from src.clustering.Clustering import Clustering


class FuzzyCMeans(Clustering):
    def __init__(self, c: int = 3, m: float = 2, max_iterations: int = 100, tolerance: float = 1e-4):
        """
        Initialize the Fuzzy C-Means clustering algorithm with the input data.

        Args:
            c (int): The number of clusters to form.
            m (float): Weighting exponent.
            max_iterations (int): The maximum number of iterations to run.
            tolerance (float): The tolerance to determine convergence.
        """
        self.c = c
        self.m = m
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.u = None
        self.centroids = None
        self.labels_ = None

        np.random.seed(0)

    def initialize_membership_matrix(self, n_samples: int) -> None:
        """
        Initialize the membership matrix with random values. The sum of each column
            should be equal to 1.

        Args:
            n_samples (int): The number of samples in the input data.
        """
        # Generate a random matrix with values between 0 and 1
        random_matrix = np.random.rand(self.c, n_samples)

        # Normalize each column, so that the sum of each column is equal to 1
        normalized_matrix = random_matrix / random_matrix.sum(axis=0)
        self.u = normalized_matrix

    def update_centroids(self, data: np.ndarray) -> np.ndarray:
        """
        Update the centroids matrix.

        Args:
            data (np.ndarray): The input data.
            u (np.ndarray): The membership matrix.
        """
        numerator = np.matmul(self.u**self.m, data)
        denominator = np.sum(self.u**self.m, axis=1)

        new_centroids = numerator / denominator[:, np.newaxis]
        
        return new_centroids

    def calculate_membership(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the membership matrix.

        Args:
            data (np.ndarray): The input data.
            centroids (np.ndarray): The centroids matrix.
        """
        data = data.astype(float)
        self.centroids = self.centroids.astype(float)

        # Calculate the Euclidean distances between data points and centroids
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        
        # Replace distances of 0 by a small value to avoid divisions by 0.
        distances = np.where(distances == 0, 1e-3, distances)

        # Compute the membership matrix in a vectorized manner
        u = (distances[:, :, np.newaxis] / distances[:, np.newaxis, :])**(2 / (self.m - 1))
        u = 1.0 / np.sum(u, axis=2)

        return u.T

    def fit(self, data: pd.DataFrame()):
        """
        Abstract method for fitting the clustering algorithm to the data,
            and assigning a class to each data point.

        Args:
            data (pandas.DataFrame): The input data for fitting the clustering
                algorithm.
        """
        data_copied = data.copy()

        # Convert the input data to a numpy array
        X = data_copied.to_numpy()
        n_samples, _ = X.shape
        self.initialize_membership_matrix(n_samples)

        # Initialize centroids to random points
        self.centroids = np.array(random.sample(list(X), self.c))

        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()

            self.centroids = self.update_centroids(X)
            self.u = self.calculate_membership(X)

            # Check for convergence
            if np.linalg.norm(self.centroids - old_centroids) < self.tolerance:
                break

        labels = np.argmax(self.u, axis=0)
        self.labels_ = labels
    
    def fit_predict(self, data: pd.DataFrame()) -> pd.DataFrame():
        """
        Abstract method for fitting the clustering algorithm to the data,
            and assigning a class to each data point.

        Args:
            data (pandas.DataFrame): The input data for fitting the clustering
                algorithm.
        """
        data_copied = data.copy()

        # Convert the input data to a numpy array
        X = data_copied.to_numpy()
        n_samples, _ = X.shape
        self.initialize_membership_matrix(n_samples)

        # Initialize centroids to random points
        self.centroids = np.array(random.sample(list(X), self.c))

        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()

            self.centroids = self.update_centroids(X)
            self.u = self.calculate_membership(X)

            # Check for convergence
            if np.linalg.norm(self.centroids - old_centroids) < self.tolerance:
                break

        labels = np.argmax(self.u, axis=0)
        self.labels_ = labels
        data_copied["label"] = labels

        return data_copied
    
    def get_membership_matrix(self):
        return self.u


if __name__ == "__main__":
    from src.read.processing import Processing

    dataclass = Processing(source_path='input/datasets/')
    df = dataclass.read('glass')
    df = df.iloc[:, :-1]
    model = FuzzyCMeans(c=3, m=2)
    df_out = model.fit_predict(df)


    print(df_out.head())
    data = df.to_numpy()

    tsne = TSNE(n_components=2, random_state=0)
    print(np.append(data, model.centroids, axis=0).shape)
    tsne_embeddings = tsne.fit_transform(np.append(data, model.centroids, axis=0))

    # Visualize clusters
    plt.scatter(tsne_embeddings[:len(data), 0], tsne_embeddings[:len(data), 1], c=np.argmax(model.u, axis=0))

    # Visualize centroids
    plt.scatter(tsne_embeddings[len(data):, 0], tsne_embeddings[len(data):, 1], c='r', marker='x')

    plt.legend(['Data', 'Centroids'])

    plt.show()