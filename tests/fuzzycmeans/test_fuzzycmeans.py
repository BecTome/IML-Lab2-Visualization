import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.clustering.FuzzyCMeans import FuzzyCMeans


@pytest.fixture
def sample_data():
    data = np.array([[1, 2], [1.5, 1.8], [1, 0.6], [5, 8], [8, 8], [9, 11]])
    df = pd.DataFrame(data, columns=["X", "Y"])
    return df

def test_fuzzy_cmeans_clustering(sample_data):
    fcm = FuzzyCMeans(c=2, m=2)
    df_result = fcm.fit_predict(sample_data)

    # Assert that the first 3 data points belong to the same cluster,
    # and the last 3 belong to the same cluster
    assert df_result.iloc[0]["label"] == df_result.iloc[1]["label"] == df_result.iloc[2]["label"]
    assert df_result.iloc[3]["label"] == df_result.iloc[4]["label"] == df_result.iloc[5]["label"]
    assert df_result.iloc[0]["label"] != df_result.iloc[3]["label"]

    # Visualize clusters
    plt.scatter(df_result['X'], df_result['Y'], c=df_result['label'])

    # Visualize centroids
    plt.scatter(fcm.centroids[:, 0], fcm.centroids[:, 1], c='r', marker='x')
    plt.legend(['Data', 'Centroids'])
    plt.show()
