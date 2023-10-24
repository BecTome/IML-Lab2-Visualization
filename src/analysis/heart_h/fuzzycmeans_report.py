from src.trainflow.TrainFlow import TrainFlow
from src.clustering.FuzzyCMeans import FuzzyCMeans
import os


ds_name = "heart-h"
model = FuzzyCMeans
target = "num"
ls_metrics = ['adjusted_rand_score', 'v_measure_score', 
                'silhouette_score', 'calinski_harabasz_score']
output_path = 'report'

d_labels = {
    0: '<50',
    1: '>50_1',
}


# Find the best hyperparameters.
tf = TrainFlow(ds_name=ds_name, model=model, target=target, metrics=ls_metrics, 
               d_plot_params=d_labels, output_path=output_path)
tf.run()

# Store the resulting dataframe with the labels.
tf.df_results.to_csv(os.path.join(tf.output_path_total, 'benchmark.csv'), index=False)

# Run TrainFlow with the best hyperparameters found above.
param_keys = [k for k in tf.training_params.keys()]
d_hyper_euclidean = {k: v for k,v in tf.df_results.to_dict('records')[0].items() 
                     if k in param_keys}

tf_best_model = TrainFlow(ds_name=ds_name, model=model, hyperparams=d_hyper_euclidean, 
                             target=target, metrics=ls_metrics, d_plot_params=d_labels,
                             output_path=output_path)
tf_best_model.run()