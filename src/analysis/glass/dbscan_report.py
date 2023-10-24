import pandas as pd
from src.trainflow.TrainFlow import TrainFlow
from src.clustering.DBSCAN import DBSCAN
import matplotlib.pyplot as plt
import os


ds_name = "glass"
model = DBSCAN
target = "Type"
ls_metrics = ['adjusted_rand_score', 'v_measure_score', 
                'silhouette_score', 'calinski_harabasz_score']

d_labels = {
            0: 'build_wind_float',
            1: 'vehic_wind_float',
            2: 'tableware',
            3: 'build_wind_non-float',
            4: 'headlamps',
            5: 'containers'
            }


tf = TrainFlow(ds_name, model, target=target, metrics=ls_metrics, 
               d_plot_params=d_labels)
tf.run()



def filter_alg_met(df, alg, met, ls_metrics):
    return df[(df['algorithm'] == alg)&(df['metric'] == met)].iloc[[0], :]

res_euclidean_auto = filter_alg_met(tf.df_results, 'auto', 'euclidean', ls_metrics)
res_euclidean_ball_tree = filter_alg_met(tf.df_results, 'ball_tree', 'euclidean', ls_metrics)
res_euclidean_kd_tree = filter_alg_met(tf.df_results, 'kd_tree', 'euclidean', ls_metrics)
res_euclidean_brute = filter_alg_met(tf.df_results, 'brute', 'euclidean', ls_metrics)

# res_manhattan_auto = filter_alg_met(tf.df_results, 'auto', 'manhattan', ls_metrics)
df_euclidean = pd.concat([res_euclidean_auto, 
                           res_euclidean_ball_tree,
                           res_euclidean_kd_tree,
                           res_euclidean_brute], axis=0)
df_euclidean.to_csv(os.path.join(tf.output_path_total, 'benchmark_euclidean.csv'), index=False)

res_cosine_auto = filter_alg_met(tf.df_results, 'auto', 'cosine', ls_metrics)
res_cosine_ball_tree = filter_alg_met(tf.df_results, 'ball_tree', 'cosine', ls_metrics)
res_cosine_kd_tree = filter_alg_met(tf.df_results, 'kd_tree', 'cosine', ls_metrics)
res_cosine_brute = filter_alg_met(tf.df_results, 'brute', 'cosine', ls_metrics)

# res_manhattan_auto = filter_alg_met(tf.df_results, 'auto', 'manhattan', ls_metrics)
df_cosine = pd.concat([res_cosine_auto, 
                           res_cosine_ball_tree,
                           res_cosine_kd_tree,
                           res_cosine_brute], axis=0)

df_cosine.to_csv(os.path.join(tf.output_path_total, 'benchmark_cosine.csv'), index=False)

param_keys = [k for k in tf.training_params.keys()]
d_hyper_euclidean = {k: v for k,v in res_euclidean_auto.to_dict('records')[0].items() 
                     if k in param_keys}

tf_euclidean = TrainFlow(ds_name=ds_name, model=model, hyperparams=d_hyper_euclidean, 
                             target=target, metrics=ls_metrics, d_plot_params=d_labels)
tf_euclidean.run()

d_hyper_cosine = {k: v for k,v in res_cosine_auto.to_dict('records')[0].items() 
                     if k in param_keys}

tf_cosine = TrainFlow(ds_name=ds_name, model=model, hyperparams=d_hyper_cosine, 
                      target=target, metrics=ls_metrics, d_plot_params=d_labels)
tf_cosine.run()

