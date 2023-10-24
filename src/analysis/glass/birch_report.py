from src.trainflow.TrainFlow import TrainFlow
from src.clustering.Birch import Birch


ds_name = "glass"
model = Birch
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
