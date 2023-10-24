from src.clustering.Clustering import Clustering

import numpy as np
import pandas as pd

class PAM(Clustering):
    def __init__(self, k: int = 3, max_iterations: int  = 20, distance_type: str = 'euclidian'):
        self.k = k
        self.max_iterations = max_iterations
        self.distance_type = distance_type
        self.clusters = None
        self.medoids = None
        self.labels_ = None
    
    def euclidean_distance(self, p1, p2):
        """
        Calculates the Euclidean distance between two points.

        Args:
            p1: A list or tuple of coordinates for the first point.
            p2: A list or tuple of coordinates for the second point.

        Returns:
            The Euclidean distance between the two points.
        """

        # Check if the points have the same number of dimensions.
        if len(p1) != len(p2):
            raise ValueError("The points must have the same number of dimensions.")

        # Calculate the Euclidean distance.
        distance = 0
        for i in range(len(p1)):
            distance += (p1[i] - p2[i])**2

        return np.sqrt(distance)


    def manhattan_distance(self, p1, p2):
        """
        Calculates the Manhattan distance between two points.

        Args:
            p1: A list or tuple of coordinates for the first point.
            p2: A list or tuple of coordinates for the second point.

        Returns:
            The Manhattan distance between the two points.
        """

        # Check if the points have the same number of dimensions.
        if len(p1) != len(p2):
            raise ValueError("The points must have the same number of dimensions.")

        # Calculate the Manhattan distance.
        distance = 0
        for i in range(len(p1)):
            distance += abs(p1[i] - p2[i])

        return distance


    def cosine_similarity(self, p1, p2):
        """
        Calculates the cosine similarity between two vectors.

        Args:
            p1: A list or tuple of coordinates for the first vector.
            p2: A list or tuple of coordinates for the second vector.

        Returns:
            The cosine similarity between the two vectors.
        """

        # Check if the vectors have the same number of dimensions.
        if len(p1) != len(p2):
            raise ValueError("The vectors must have the same number of dimensions.")

        # Calculate the dot product of the two vectors.
        dot_product = 0
        for i in range(len(p1)):
            dot_product += p1[i] * p2[i]

        # Calculate the cosine similarity.
        cosine_similarity = dot_product / (np.linalg.norm(p1) * np.linalg.norm(p2))

        return cosine_similarity
        
    def pam(self, data: np.array) -> (np.array, np.array, float):
        # Initialize variable of n as the number of columns (dimensions)
        n, _ = data.shape

        # Randomly initialize medoids
        medoids = np.array(data[np.random.choice(n, self.k, replace=False)])

        distance = self.euclidean_distance
        if self.distance_type == 'manhattan':
            distance = self.manhattan_distance
        elif self.distance_type == 'cosine':
            distance = self.cosine_similarity

        # Assign each data point to the closest medoid
        clusters = np.argmin(np.apply_along_axis(lambda x: np.apply_along_axis(distance, 1, medoids, x), 1, data), axis=1)

        # Calculate the total cost of the current clustering
        total_cost = sum(distance(data[i], medoids[clusters[i]]) for i in range(n))

        for _ in range(self.max_iterations):

            # Swap medoids and reassign if it reduces the total cost
            for i in range(self.k):
                temp_medoids = np.copy(medoids)
                candidate_medoids = np.where(clusters == i)[0]
                for candidate in candidate_medoids:
                    temp_medoids[i] = data[candidate]
                    new_clusters = np.argmin(np.apply_along_axis(lambda x: np.apply_along_axis(distance, 1, temp_medoids, x), 1, data), axis=1)
                    new_cost = sum(distance(data[j], temp_medoids[new_clusters[j]]) for j in range(n))
                    if new_cost < total_cost:
                        medoids = np.copy(temp_medoids)
                        clusters = np.copy(new_clusters)
                        total_cost = new_cost

        return clusters, medoids, total_cost

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
        clusters, _, _ = self.pam(X)
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
        clusters, medoids, _ = self.pam(X)
        
        self.clusters = clusters
        self.medoids = medoids

        data["label"] = clusters
        self.labels_ = clusters
        return data