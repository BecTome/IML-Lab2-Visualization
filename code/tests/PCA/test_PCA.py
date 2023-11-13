import numpy as np
from src.decomposition.PCA import PCA

from sklearn.decomposition import PCA as PCAsk

def test_PCA_fit_transform():
    # Test the fit_transform method
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    pca = PCA(n_components=1)
    transformed_X = pca.fit_transform(X)
    expected_transformed_X = PCAsk(1).fit_transform(X)

    print(transformed_X, expected_transformed_X)
    print(pca.components_)
    assert np.allclose(transformed_X, expected_transformed_X)

def test_PCA_transform():
    # Test the transform method
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    pca = PCA(n_components=1)
    pca.fit(X)
    transformed_X = pca.transform(X)
    expected_transformed_X = PCAsk(1).fit_transform(X)

    assert np.allclose(transformed_X, expected_transformed_X)
