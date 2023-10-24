from src.read.processing import Processing
from src.evaluation.evaluator import Evaluator
import config
import os
import datetime
import json
import pickle
# Set a logger with timestamp
import matplotlib.pyplot as plt
import numpy as np
# Create a logger

from src.utils.utils import header
from src.utils.logger import setup_logger


class TrainFlow:
    '''
    Class which encapsulates the whole process of training and evaluating a model
    '''

    def __init__(self, 
                 ds_name: str, 
                 model,  
                 metrics: list,
                 target:str,
                 d_plot_params: dict,
                 output_path: str=config.OUTPUT_PATH,
                 top_k: int=5,
                 hyperparams: dict=None) -> None:
        '''
        Inputs:
            * ds_name: name of the dataset
            * model: model to train (not instantiated)
            * hyperparams: dictionary with the initial hyperparameters to tune 
                            (e.g. {'threshold': .5,'n_clusters': 6})
            * metrics: list of metrics to use for the evaluation
            * d_plot_params: dictionary with the parameters to plot the results
            * output_path: path where the results will be saved
            * top_k: number of top hyperparameters to save
        '''

        # Initialization Attributes
        self.ds_name = ds_name
        self.model = model
        self.metrics = metrics
        self.output_path = output_path
        self.top_k = top_k
        self.d_plot_params = d_plot_params
        self.hyperparams = hyperparams
        self.target = target

        # Attributes to be filled during the execution
        self.output_path_total = None
        self.data_loader = None
        self.d_best_params = None
        self.training_params = None
        self.trained_model = None
        self.trained_model_scores = None
        self.evaluator = None
        self.df_results = None

        self.logger = setup_logger('trainflow.log')
        
        #### Load Data ####
        self.logger.info(header("Loading data"))
        self.load_data()
        self.logger.info(f"Data loaded: {self.ds_name}")

        

    def run(self):
        '''
        Run step by step the whole process of training and evaluating a model
        Write the results in the output folder
        '''
        #### Initialization ####
        self.logger.info(f"Running {self.model.__name__} on {self.ds_name} dataset")

        # Set up the evaluator for the dataset
        self.evaluator = Evaluator(self.model, self.data_loader, self.target)

        #### Tune Hyperparams ####
        if self.hyperparams is None:
            self.logger.info(header("Tuning hyperparameters"))
            self.tune_hyperparams()

        #### Train Model ####
        self.logger.info(header("Training model"))
        self.train_model()
        self.logger.info(f"Trained with: {self.training_params}")

        #### Evaluate Model ####
        self.logger.info(header("Evaluating the model"))
        self.evaluate_model()
        self.logger.info(f"Scores: {self.trained_model_scores}")

        #### Write Results ####
        self.logger.info(header(f"Writing the results"))
        self.write_results()
        self.logger.info(f"Results written in {self.output_path_total}")

        #### Done ####
        self.logger.info(header(f"Done!"))


    def load_data(self):
        '''
        Creates a Processing object and reads the data
        Builds the evaluator for the dataset and the model
        '''
        # Read Data
        data_loader = Processing()
        data_loader.read(self.ds_name)

        # Apply the corresponding preprocessing
        data_loader.general_preprocessing()
        self.data_loader = data_loader

    def tune_hyperparams(self):
        '''
        Tune the hyperparameters of the model using the evaluator        
        '''

        # Get the results for each combination of hyperparameters
        # and sort them by the metrics
        df_results = self.evaluator.get_results_hyperparams()
        asc = [False for _ in self.metrics]
        df_results = df_results.sort_values(by=self.metrics, ascending=asc)

        # Get the best hyperparameters and write themd_best_params
        model_name = self.model.__name__
        d_best_params = self.evaluator.get_best_hyperparams(self.metrics[0])
        # Remove the metrics and keep just the hyperparameters
        d_best_params = {k:v for k,v in d_best_params.items() if k in\
                              self.evaluator.d_hyperparams[model_name].keys()}

        self.d_best_params = d_best_params
        self.df_results = df_results


    def train_model(self):
        '''
        Train the model
        If some hyperparameters are given, train the model with them
        Otherwise, train the model with the best hyperparameters
        '''
        # Train the model with the best hyperparameters or with the hyperparameters given
        if self.hyperparams is None:
            # Get the best hyperparameters
            self.training_params = self.d_best_params
        else:
            self.training_params = self.hyperparams

        # Train the model with the best hyperparameters
        self.trained_model = self.model(**self.training_params)
        #self.trained_model_predict(self.data_loader.df.drop(self.target, axis=1))
        self.trained_model.fit_predict(self.data_loader.df.drop(self.target, axis=1))

    def evaluate_model(self):
        '''
        Evaluate the model with the best hyperparameters
        '''
        self.trained_model_scores = self.evaluator.score_hyperparams(**self.training_params)

    def write_results(self):
        '''
        Write the results in the output folder
        '''
        # Create the output folder
        self.create_output_folder()   

        # Write the results of the hyperparameters tuning
        if self.hyperparams is None: 
            # Total hyperopt results
            filename = f"{self.output_path_total}/results.csv"
            self.df_results.to_csv(filename, index=False)

            # Top k hyperopt results
            for metric in self.metrics:
                df_top_k_best = self.evaluator.get_top_k_hyperparams(metric, self.top_k)
                filename = f"{self.output_path_total}/top_{self.top_k}_{metric}.csv"
                df_top_k_best.to_csv(filename, index=False)
            
        # TRaining hyperparameters
        main_metric = self.metrics[0]
        best_metric = self.trained_model_scores[main_metric]
        filename = f"{self.output_path_total}/best_hyperparams_{main_metric}_{best_metric}.json"
        with open(filename, "w") as f:
            json.dump(str(self.training_params), f)

        # Format folder names with the scores
        internal_metrics = [x.__name__ for x in self.evaluator.d_metrics["internal"]]
        external_metrics = [x.__name__ for x in self.evaluator.d_metrics["external"]]
        scores_str = {k:v for k,v in self.trained_model_scores.items() if k in\
                        internal_metrics + external_metrics}
        str_metrics = "_".join([f"{key}_{value:.4f}" for key, value in scores_str.items()
                                if value is not None])
        
        # Save the model
        filename = f"{self.output_path_total}/model_{str_metrics}_.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

        # Copy and move the log file
        #filename = f"{self.output_path_total}/trainflow.log"
        #os.rename("trainflow.log", filename)

        # Plot and save the results
        img_path = os.path.join(self.output_path_total, 'img')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        filename = f"{img_path}/{self.model.__name__}_{self.ds_name}_scatter.jpg"
        self.evaluator.plot_scatter_pca_visualization(self.trained_model.labels_, 
                                                      d_classes = self.d_plot_params)

        plt.savefig(filename)

        
    def create_output_folder(self):
        # Convert the hyperparameters to a string (e.g., "param1_value1_param2_value2_param3_value3")
        hyperparams_str = "_".join([f"{key}_{value}" for key, value in self.training_params.items()])

        # Get the current time as a string
        current_date = datetime.datetime.now().strftime("%d-%m-%Y_%Hh-%Mm-%Ss")

        # Create the folder name using hyperparameters and the date
        folder_name = f"{hyperparams_str}/{current_date}"

        # Create the full folder path
        output_path_total = f"{self.output_path}/{self.ds_name}/{self.model.__name__}/"
        output_path_total = os.path.join(output_path_total, folder_name)

        # Check if the folder already exists and create it if not
        if not os.path.exists(output_path_total):
            os.makedirs(output_path_total)

        self.output_path_total = output_path_total
            