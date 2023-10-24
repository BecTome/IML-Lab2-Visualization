from src.trainflow.TrainFlow import TrainFlow
from src.clustering.Birch import Birch


ds_name = "vote"
model = Birch
target = "Class"
ls_metrics = ['adjusted_rand_score', 'v_measure_score', 
                'silhouette_score', 'calinski_harabasz_score']

d_labels = {
    0: 'democrat',
    1: 'republican',
}


tf = TrainFlow(ds_name, model, target=target, metrics=ls_metrics, 
               d_plot_params=d_labels)
tf.run()
