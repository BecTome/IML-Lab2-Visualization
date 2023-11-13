from abc import ABC, abstractmethod

import pandas as pd


class Clustering(ABC):
    def __init__(self) -> None:
        """
        Initialize the clustering algorithm with the input data.
        """
        pass

    @abstractmethod
    def fit_predict(self, data: pd.DataFrame()) -> pd.DataFrame():
        """
        Abstract method for fitting the clustering algorithm to the data,
            and assigning a class to each data point.

        Args:
            data (pandas.DataFrame): The input data for fitting the clustering
                algorithm.
        """
        pass
