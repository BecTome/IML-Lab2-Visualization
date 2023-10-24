from src.clustering.CLARA import PAM

import pandas as pd
import numpy as np
import pytest

def test_pam():
    map_data = {
        'x1': [0, 0.5, 1, 0, 0.5, 1],
        'x2': [0, 0, 0, 5, 5, 5],
    }
    data = np.array([[0, 0], [0.5, 0], [1, 0],
            [0, 5], [0.5, 5], [1, 5]])
    
    clusters, medoids = PAM.pam(data, 2, max_iterations=5)

    assert (
        np.array_equal(clusters, [1, 1, 1, 0, 0, 0])
        | np.array_equal(clusters, [0, 0, 0, 1, 1, 1])
    )
    assert (
        np.array_equal(medoids, [[0.5, 5], [0.5, 0]])
        | np.array_equal(medoids, [[0.5, 0], [0.5, 5]])
    )