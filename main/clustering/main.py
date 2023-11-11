from src.trainflow.TrainFlow import TrainFlow
import config
from argparse import ArgumentParser
from src.clustering.KMeans import KMeans
from src.clustering.Birch import Birch

def standardize_input (inp: str = ""):
    return inp.lower().replace('-')

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="write report to FILE", default='glass')
parser.add_argument("-a", "--algorithm",
                    action="algorithm", dest="verbose", default='kmeans',
                    help="don't print status messages to stdout")

args = parser.parse_args()

dataset = standardize_input(args['dataset'])
algorithm = standardize_input(args['algorithm'])

if algorithm == 'kmeans':
    model = KMeans
elif algorithm == 'birch':
    model = Birch
else:
    raise Exception('Algorithm not found')

dataset_config = config.D_FEATURES[dataset]
target = str()
d_labels_aux = dataset_config['nominal_ordinal']

if 'num' in d_labels_aux:
    d_labels = d_labels_aux['num']

d_labels = {}
for i, key in enumerate(d_labels_aux.keys()):
    d_labels[i] = key

LS_METRICS = ['adjusted_rand_score', 'v_measure_score', 
                'silhouette_score', 'calinski_harabasz_score']

tf = TrainFlow(dataset, model, target=target, metrics=LS_METRICS, 
               d_plot_params=d_labels)
tf.run()
