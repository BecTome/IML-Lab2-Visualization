import pandas as pd
from src.clustering.Clustering import Clustering
from sklearn.cluster import Birch as skBirch

class Birch(skBirch):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit_predict(self, data: pd.DataFrame()) -> pd.DataFrame():
            """
            Abstract method for fitting the clustering algorithm to the data,
                and assigning a class to each data point.

            Args:
                data (pandas.DataFrame): The input data for fitting the clustering
                    algorithm.
            """
            # Convert the input data to a numpy array
            X = data.copy()
            X = X.to_numpy()

            self.fit(X)

            labels = self.labels_
            data["label"] = labels
            return data
    

if __name__ == "__main__":
     from src.read.processing import Processing

     dataclass = Processing(source_path='input/datasets/')
     df = dataclass.read('glass')
     dataclass.general_preprocessing()
     model = Birch()
     df_out = model.fit_predict(dataclass.df)
        
     print(df_out.head())