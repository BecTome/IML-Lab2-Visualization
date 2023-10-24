import pytest
import pandas as pd
import numpy as np

from src.evaluation.evaluator import Evaluator  # Import the Evaluator class from your module
from src.read.processing import Processing
from src.clustering.DBSCAN import DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score,\
                            v_measure_score

d_metrics = {'internal': [silhouette_score], #, 'davies_bouldin', 'calinski_harabasz'],
            'external': [homogeneity_score, completeness_score, v_measure_score]}

d_hyperparams = {
                    'DBSCAN': {
                            'eps': np.arange(0.1, 1.0, 0.1),
                            'min_samples': np.arange(2, 10, 1)
                            }
                }
thresh_cat = 10 # Limit of categories for one hot encoding
d_features = {
                "glass":
                        {
                            "numeric":["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],
                            "nominal": [],
                            "nominal_ordinal": {
                                                "Type": {
                                                        'build_wind_float': 0,
                                                        'vehic_wind_float': 1,
                                                        'tableware': 2,
                                                        'build_wind_non-float': 3,
                                                        'headlamps': 4,
                                                        'containers': 5
                                                        },
                                                }
                        }
                }

# Define a fixture to create an instance of the Evaluator class for testing
@pytest.fixture
def evaluator_instance():
    # You need to create a proper data_loader instance and pass it here
    # For now, we'll use a placeholder None.

    data_loader = Processing(d_features=d_features, thresh_cat=thresh_cat)
    data_loader.read("glass")

    return Evaluator(model=DBSCAN, data_loader=data_loader, target="Type",\
                     d_metrics=d_metrics, d_hyperparams=d_hyperparams)

def test_get_results_hyperparams(evaluator_instance):
    # Test that get_results_hyperparams returns a DataFrame
    df_results = evaluator_instance.get_results_hyperparams()
    assert isinstance(df_results, pd.DataFrame)

def test_score_hyperparams(evaluator_instance):
    # Test that score_hyperparams returns a dictionary with scores
    kw_hyperparams = {"eps": .5, "min_samples": 6}  # Replace with actual hyperparameters
    d_scores = evaluator_instance.score_hyperparams(**kw_hyperparams)
    assert isinstance(d_scores, dict)

def test_grid_from_dict():
    # Test grid_from_dict function
    d_in = {"param1": [1, 2], "param2": ["a", "b"]}
    ls_hyps = Evaluator.grid_from_dict(d_in)
    
    # You should make assertions about the expected combinations
    # For example, if d_in has 2 parameters with 2 values each, there should be 4 combinations.
    assert len(ls_hyps) == 4

# You can add more test cases as needed
