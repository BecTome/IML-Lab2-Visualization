import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KModes:

    """Class to perform K-Modes Clustering"""
    
    def __init__(self, k=3, random_state=0):
        """Initialize the KModesClustering class"""
        self.k = k
        self.centroids = None
        self.labels_ = None
        self.random_state = random_state

    @staticmethod
    def matching_distance(a, b):
        """Calculate the matching distance between a data point and the centroids"""
        return np.sum(a != b)
    
    @staticmethod
    def find_mode(data):
        """Find the mode of each column in the data"""
        mode = []
        for i in range(data.shape[1]):
            unique_values, counts = np.unique(data[:, i], return_counts=True)
            mode_value = unique_values[np.argmax(counts)]
            mode.append(mode_value)

        return np.array(mode)
    
    def fit_predict(self, data, max_iterations=100):
        """Train the K-Modes Clustering model"""
        np.random.seed(self.random_state)
        # Convert the input data to a numpy array
        X = data.copy()
        X = data.to_numpy()
        
        # Initialize random data points to centroids
        self.centroids = np.array(X[np.random.choice(X.shape[0], self.k, replace=False)])
        
        for _ in range(max_iterations):
            idxs = []

            # Assign each data point to the closest centroid
            for data_point in X:
                distances = [self.matching_distance(data_point, centroid) for centroid in self.centroids]
                cluster_number = np.argmin(distances)
                idxs.append(cluster_number)

            idxs = np.array(idxs)

            new_centroids = []

            for i in range(self.k):
                cluster_mode = self.find_mode(X[idxs == i])
                new_centroids.append(cluster_mode)

            new_centroids = np.array(new_centroids)

            # Check for convergence
            if np.array_equal(self.centroids, new_centroids):
                break
            else:
                self.centroids = new_centroids

        # Convert idxs to a DataFrame column
        data["label"] = idxs
        self.labels_ = idxs
        return data

if __name__ == "__main__":
    from src.read.processing import Processing

    dataclass = Processing(source_path='input/datasets/')
    df = dataclass.read('connect-4')
    df = df.iloc[:, :-1]
    kmodes =KModes(k=3)
    df_out = kmodes.fit_predict(df)

    print(df_out.head())
    data = df_out.to_numpy()
    plt.scatter(data[:, 0], data[:, 1], c=df_out["label"])
    plt.scatter(kmodes.centroids[:, 0], kmodes.centroids[:, 1], c='r', marker='x')
    plt.show()

