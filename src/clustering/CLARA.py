from src.clustering.Clustering import Clustering
from src.clustering.PAM import PAM

import numpy as np
import pandas as pd

class CLARA(Clustering):
    def __init__(self, k: int = 3, n: int = 100, max_iterations: int = 5, distance_type: str = 'euclidian'):
        self.k = k
        self.n = n
        self.max_iterations = max_iterations
        self.distance_type = distance_type
        self.labels_ = None
        self.pam = PAM(self.k, self.max_iterations, self.distance_type)

    def clara(self, data: pd.DataFrame) -> (np.array, np.array, float):
        best_cost = float('inf')
        best_clusters = None
        
        for _ in range(self.max_iterations):
            sample_indices = np.random.choice(len(data), self.n, replace=False)
            sample = data[sample_indices]
            
            # Apply K-Medoids on the sample
            clusters, medoids_indices, cost = self.pam.pam(sample)
            
            if cost < best_cost:
                best_cost = cost
                best_clusters = clusters
        
        return best_clusters, medoids_indices, best_cost
    
    def fit(self, data: pd.DataFrame()) -> pd.DataFrame():
        """
        Abstract method for fitting the clustering algorithm to the data,
            and assigning a class to each data point.

        Args:
            data (pandas.DataFrame): The input data for fitting the clustering
                algorithm.
        """
        # Convert the input data to a numpy array
        X = data.to_numpy()
        clusters, _, _ = self.clara(X)
        self.labels_ = clusters
    
    def fit_predict(self, data: pd.DataFrame()) -> pd.DataFrame():
        """
        Abstract method for fitting the clustering algorithm to the data,
            and assigning a class to each data point.

        Args:
            data (pandas.DataFrame): The input data for fitting the clustering
                algorithm.
        """
        # Convert the input data to a numpy array
        X = data.to_numpy()
        clusters, _, _ = self.clara(X)
        self.labels_ = clusters
        data["label"] = clusters
        return data
