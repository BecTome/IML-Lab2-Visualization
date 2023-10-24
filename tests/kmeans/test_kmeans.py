import unittest
import numpy as np
from src.clustering.KMeans import KMeans
import pandas as pd

class TestKMeans(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.data = pd.DataFrame(np.random.rand(100, 2), columns=["X", "Y"])
        self.kmeans = KMeans(k=2)
    
    def test_kmeans_fit_predict(self):
        df_out, cluster_idxs = self.kmeans.fit_predict(self.data)
        self.assertEqual(len(cluster_idxs), len(self.data))

if __name__ == '__main__':
    unittest.main()
