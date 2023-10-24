import pandas as pd
from src.trainflow.TrainFlow import TrainFlow
from src.clustering.PAM import PAM
import matplotlib.pyplot as plt
import os

ds_name = "heart-h"
model = PAM
target = "num"
ls_metrics = ['adjusted_rand_score', 'v_measure_score', 
                'silhouette_score', 'calinski_harabasz_score']
output_path = 'report'
d_labels = {
            0: '<50',
            1: '>50_1',
            }


tf = TrainFlow(ds_name, model, target=target, metrics=ls_metrics, 
               d_plot_params=d_labels, output_path=output_path)
tf.run()

def filter_alg_met(df, met):
    return df[df['distance_type'] == met].iloc[[0], :]

res_euclidean = filter_alg_met(tf.df_results, 'euclidean')
res_manhattan = filter_alg_met(tf.df_results, 'mahattan')
res_cosine = filter_alg_met(tf.df_results, 'cosine')


df_results = pd.concat([res_euclidean, 
                           res_manhattan,
                           res_cosine], axis=0)

df_results.to_csv(os.path.join(tf.output_path_total, 'benchmarks.csv'), index=False)

param_keys = [k for k in tf.training_params.keys()]
d_hyper_euclidean = {k: v for k,v in res_euclidean.to_dict('records')[0].items() 
                     if k in param_keys}

tf_euclidean = TrainFlow(ds_name=ds_name, model=model, hyperparams=d_hyper_euclidean, 
                             target=target, metrics=ls_metrics, d_plot_params=d_labels,
                             output_path=output_path)
tf_euclidean.run()

d_hyper_manhattan = {k: v for k,v in res_manhattan.to_dict('records')[0].items() 
                     if k in param_keys}

tf_manhattan = TrainFlow(ds_name=ds_name, model=model, hyperparams=d_hyper_manhattan, 
                      target=target, metrics=ls_metrics, d_plot_params=d_labels,
                      output_path=output_path)
tf_manhattan.run()


d_hyper_cosine = {k: v for k,v in res_cosine.to_dict('records')[0].items() 
                     if k in param_keys}

tf_cosine = TrainFlow(ds_name=ds_name, model=model, hyperparams=d_hyper_cosine, 
                      target=target, metrics=ls_metrics, d_plot_params=d_labels,
                      output_path=output_path)
tf_cosine.run()

