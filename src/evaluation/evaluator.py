import config
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder



class Evaluator:

    def __init__(self, 
                 model, 
                 data_loader, 
                 target: str, 
                 d_metrics: dict=config.D_METRICS,
                 d_hyperparams: dict=config.D_HYPERPARAMS):
        '''
        Input:
            model: class of the model to be evaluated 
                    (not instantiated, e.g. KMeans not KMeans())
            data_loader: instance of the class Processing
            target: name of the target column
            d_metrics: dictionary with the metrics to be used
            d_hyperparams: dictionary with the hyperparameters to be used

        Attributes:
            model, target, d_metrics, d_hyperparams
            
            data: dataframe with the data from data_loader
            y_true: true labels (target column from data)
        '''
        self.model = model
        self.data = data_loader.df
        self.target = target
        self.y_true = self.data[self.target]
        self.d_metrics = d_metrics
        self.d_hyperparams = d_hyperparams
        self.df_results = None

        self.pca = PCA(n_components=2)
        # Delete the target column from the data to avoid use it as a feature
        self.pc = self.pca.fit_transform(self.data.drop(columns=self.target).to_numpy())

    def get_best_hyperparams(self, metric: str):
        '''
        Input:
            metric: name of the metric to be used
        Output:
            d_best: dictionary with the best hyperparameters
        '''
        # Get the best hyperparameters
        model_name = self.model.__name__
        hyperparams = self.d_hyperparams[model_name]

        d_best = self.get_top_k_hyperparams(metric, 1).to_dict('records')[0]

        d_best = {k:v for k,v in d_best.items() if k in hyperparams.keys()}

        return d_best
    
    def get_top_k_hyperparams(self, metric: str, k: int):
        '''
        Input:
            metric: name of the metric to be used
            k: number of hyperparameters to be returned

        Output:
            d_best: dictionary with the top k best hyperparameters
        '''
        # Get the results for each combination of hyperparameters
        if self.df_results is None:
            df_results = self.get_results_hyperparams()
        else:
            df_results = self.df_results

        # Get the best hyperparameters
        df_top_k = df_results.sort_values(by=metric, ascending=False).iloc[:k]

        return df_top_k
        


    def get_results_hyperparams(self):
        '''
        Returns a dataframe with the results of the evaluation
        for each combination of hyperparameters
        '''
        # Get the grid of hyperparameters
        model_name = self.model.__name__
        ls_hyps = self.grid_from_dict(self.d_hyperparams[model_name])

        # Get the scores for each combination of hyperparameters
        ls_scores = []
        for hyps in ls_hyps:
            print(hyps)
            try:
                d_scores = self.score_hyperparams(**hyps)
                ls_scores.append(d_scores)
            except:
                d_scores = hyps.copy()
                d_scores['n_out_clusters'] = None
                d_scores['n_in_classes'] = len(set(self.y_true))
                for scorer in self.d_metrics["internal"]:
                    d_scores[scorer.__name__] = None
                for scorer in self.d_metrics["external"]:
                    d_scores[scorer.__name__] = None

                ls_scores.append(d_scores)

        # Create a dataframe with the results
        df_results = pd.DataFrame(ls_scores)

        self.df_results = df_results
        return df_results


    def score_hyperparams(self, **kw_hyperparams):
        '''
        Input:
            d_hyperparams: dictionary with the hyperparameters to be used
        Output:
            d_scores: dictionary with the scores for each metric
        '''

        # Get feature matrix X
        X = self.data.drop(self.target, axis=1).copy()
        
        # Instantiate and fit the model with the hyperparameters
        model = self.model(**kw_hyperparams)
        y_pred = model.fit_predict(X)["label"]

        # Get the scores iterating over the internal and external scorers
        internal_scorers = self.d_metrics["internal"]
        external_scorers = self.d_metrics["external"]

        d_scores = kw_hyperparams.copy()
        d_scores['n_out_clusters'] = len(set(y_pred))
        d_scores['n_in_classes'] = len(set(self.y_true))
        
        for scorer in external_scorers:
            try:
                score = scorer(self.y_true, y_pred)
                d_scores[scorer.__name__] = score
            except:
                d_scores[scorer.__name__] = None
        
        for scorer in internal_scorers:
            try:
                score = scorer(X, y_pred)
                d_scores[scorer.__name__] = score
            except:
                d_scores[scorer.__name__] = None
        
        return d_scores

    @staticmethod
    def grid_from_dict(d_in):
        '''
        Receives a dictionary of lists of hyperparameters and 
        returns a list of dictionaries with all the possible combinations
        '''
        # Create a list of all possible hyperparameter combinations
        combinations = list(product(*d_in.values()))

        # Print the grid of hyperparameter combinations
        ls_hyps = []
        for combo in combinations:
            hyperparameters = dict(zip(d_in.keys(), combo))
            ls_hyps.append(hyperparameters)
        print(ls_hyps)
        return ls_hyps
    
    def plot_scatter_pca_visualization(self, clusters, 
                                       centroids: np.array = None, 
                                       d_classes: dict = None):
        
        components = pd.DataFrame(self.pc, columns=['PC1', 'PC2'])
        components[self.target] = self.data[self.target]
        components['cluster'] = clusters

        components['y_true'] = components[self.target]
        components.sort_values(by=["y_true", "cluster"], inplace=True)

        # Map each string label to a numerical one.
        label_encoder = LabelEncoder()
        label_array = label_encoder.fit_transform(components[self.target])

        if self.model.__name__ != 'DBSCAN':
            # Match the true labels with the predicted labels using the Hungarian algorithm.
            mapped_labels = self.hungarian_matching(label_array, components['cluster'])
        else:
            mapped_labels = components['cluster']

        if d_classes is not None:
            components[self.target] = components[self.target].map(d_classes)

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.scatterplot(data=components, x='PC1', y='PC2',
                        hue=self.target, palette='Set1',
                        ax=ax[0])
        ax[0].set_title('True Classes')

        sns.scatterplot(data=components, x='PC1', y='PC2',
                        hue=mapped_labels, palette='Set1',
                        ax=ax[1])
        ax[1].set_title('Clustered Classes')

        if centroids is not None:
            centroids_comp = self.pca.transform(centroids)
            sns.scatterplot(x=centroids_comp[:, 0], y=centroids_comp[:, 1],
                            palette='Set1', ax=ax[1], marker='*', s=300, c='#050505')

        exp_var = self.pca.explained_variance_ratio_.sum()
        fig.suptitle(f"PCA Visualization ({exp_var:.2%} of variance explained)")
        fig.tight_layout()
        return fig, ax

    def plot_pca(self):

        X = self.data.drop(self.target, axis=1).copy()

        fig, ax = plt.subplots()
        aux_pca = PCA(n_components=len(X.columns))
        _ = aux_pca.fit_transform(X.to_numpy())
        features = range(1, aux_pca.n_components_ + 1)
        ax.bar(features, aux_pca.explained_variance_ratio_, color='#640039')
        plt.xlabel('PCA Features')
        plt.ylabel('Variance %')
        plt.xticks(features)
        plt.title("Principal Component Analysis")
        return fig, ax

    def plot_hyperparameter_tuning(self, params, scores):
        """
        Plots a line chart of hyperparameter tuning results.

        Args:
        params: A dictionary of hyperparameters and their values.
        scores: A dictionary of hyperparameter values and their corresponding scores.
        """

        # Get the hyperparameter names and values.
        hyperparameter_names = list(params.keys())
        hyperparameter_values = list(params.values())

        # Get the hyperparameter scores.
        scores = list(scores.values())

        # Sort the hyperparameter names and values by score.
        sorted_hyperparameter_names = []
        sorted_hyperparameter_values = []
        sorted_scores = []

        for i in np.argsort(scores):
            sorted_hyperparameter_names.append(hyperparameter_names[i])
            sorted_hyperparameter_values.append(hyperparameter_values[i])
            sorted_scores.append(scores[i])

        fig, ax = plt.subplots()
        # Create the line plot.
        ax.plot(sorted_hyperparameter_values, sorted_scores)

        # Add labels and a title.
        plt.xlabel("Hyperparameter value")
        plt.ylabel("Score")
        plt.title("Hyperparameter tuning results")

        return fig, ax
    

    @staticmethod
    def hungarian_matching(true_labels, predicted_labels):
        # Create a cost matrix where the cost of matching true labels to
        # predicted labels is 1 if they match and 0 if they don't.
        num_true_labels = len(np.unique(true_labels))
        num_predicted_labels = len(np.unique(predicted_labels))
        cost_matrix = np.zeros((num_true_labels, num_predicted_labels))

        for i in range(num_true_labels):
            for j in range(num_predicted_labels):
                cost_matrix[i][j] = np.sum((true_labels == i) & (predicted_labels == j))

        # Use the Hungarian algorithm to find the optimal assignment
        # The negative sign is used for minimization
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)

        # Map the predicted labels to the true labels
        label_mapping = {}
        for i, j in zip(row_indices, col_indices):
            label_mapping[j] = i

        # Assign new labels to unmatched predicted labels
        new_label = num_true_labels
        for j in range(num_predicted_labels):
            if j not in label_mapping:
                label_mapping[j] = new_label
                new_label += 1

        # Map the predicted labels to the true labels based on the mapping
        mapped_labels = np.array([label_mapping[predicted_label] for predicted_label in predicted_labels])

        return mapped_labels