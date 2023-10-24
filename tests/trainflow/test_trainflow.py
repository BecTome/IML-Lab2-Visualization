import pytest
from src.trainflow.TrainFlow import TrainFlow  # Import the TrainFlow class from your module
from src.clustering.DBSCAN import DBSCAN
import config

# Define a fixture for creating an instance of TrainFlow for testing
@pytest.fixture
def trainflow_instance():
    # You need to create a proper Processing object and model class for the initialization.
    # For now, we'll use placeholders None.
    ds_name = "glass"
    model = DBSCAN
    metrics = [config.D_METRICS["internal"][0].__name__]
    target = "Type"
    output_path = "output/test/"
    top_k = 5
    return TrainFlow(ds_name=ds_name, model=model, metrics=metrics,\
                     target=target, output_path=output_path, top_k=top_k)

def test_create_trainflow_instance(trainflow_instance):
    # Test if a TrainFlow instance is created successfully
    assert isinstance(trainflow_instance, TrainFlow)

def test_load_data(trainflow_instance):
    # Test the load_data method
    trainflow_instance.load_data()
    assert trainflow_instance.data_loader is not None  # Check if data_loader is set

def test_tune_hyperparams(trainflow_instance):
    # Test the tune_hyperparams method
    trainflow_instance.load_data()
    trainflow_instance.tune_hyperparams()
    assert trainflow_instance.d_best_params is not None  # Check if d_best_params is set

def test_train_model(trainflow_instance):
    # Test the train_model method
    trainflow_instance.load_data()
    trainflow_instance.tune_hyperparams()
    trainflow_instance.train_model()
    assert trainflow_instance.trained_model is not None  # Check if trained_model is set

def test_evaluate_model(trainflow_instance):
    # Test the evaluate_model method
    trainflow_instance.load_data()
    trainflow_instance.tune_hyperparams()
    trainflow_instance.train_model()
    trainflow_instance.evaluate_model()
    assert trainflow_instance.trained_model_scores is not None  # Check if trained_model_scores is set


def test_create_output_folder(trainflow_instance):
    # Test the create_output_folder method
    trainflow_instance.load_data()
    trainflow_instance.tune_hyperparams()
    trainflow_instance.train_model()
    trainflow_instance.evaluate_model()
    trainflow_instance.create_output_folder()
    assert trainflow_instance.output_path_total is not None  # Check if output_path_total is set

# You can add more test cases and assertions as needed
