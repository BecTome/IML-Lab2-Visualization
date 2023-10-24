from src.trainflow.TrainFlow import TrainFlow
from src.clustering.Birch import Birch


ds_name = "heart-h"
model = Birch
target = "num"
ls_metrics = ['adjusted_rand_score', 'v_measure_score', 
                'silhouette_score', 'calinski_harabasz_score']

d_labels = {
            0:'<50',
            1:'>50_1'
            }


tf = TrainFlow(ds_name, model, target=target, metrics=ls_metrics, 
               d_plot_params=d_labels)
tf.run()
