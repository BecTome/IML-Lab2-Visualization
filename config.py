
import numpy as np
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score,\
                            v_measure_score, adjusted_rand_score, calinski_harabasz_score

## Reading
SOURCE_PATH = 'input/datasets/'
OUTPUT_PATH = 'report/'

## PREPROCESSING
# Dictionary with the features of each dataset
THRESH_CAT = 10 # Limit of categories for one hot encoding
D_FEATURES = {
	"glass":
	{
		"numeric":["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],
		"nominal": [],
		"nominal_ordinal": {
			"Type": {
				'build_wind_float': 0,
				'vehic_wind_float': 1,
				'tableware': 2,
				'build_wind_non-float': 3,
				'headlamps': 4,
				'containers': 5
			},
		}
	},
"hearth":
	{
		"numeric":["trestbps", "chol", "thalach", "oldpeak"],
		"nominal": ['sex', 'chest_pain', 'fbs', 'restecg', 
				'exang', 'slope', 'thal'],
		"nominal_ordinal": {
			"num": {
				'<50': 0,
				'>50_1': 1
			},
			"age":{}
		}
	},
"vote":
	{
		"numeric":[],
		"nominal": ['handicapped-infants', 
			'water-project-cost-sharing',
			'adoption-of-the-budget-resolution', 
			'physician-fee-freeze',
			'el-salvador-aid', 'religious-groups-in-schools',
			'anti-satellite-test-ban', 
			'aid-to-nicaraguan-contras', 'mx-missile',
			'immigration', 'synfuels-corporation-cutback', 
			'education-spending', 'superfund-right-to-sue', 
			'crime', 'duty-free-exports',
			'export-administration-act-south-africa'],
		"nominal_ordinal": {
			"Class": {
                'republican': 1,
				'democrat': 0
			},
		}
	}
}

D_TARGETS = {'glass': 'Type', 'heart-h': 'num', 'vote': 'Class'}

## Evaluation
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score,\
                            v_measure_score, adjusted_rand_score, calinski_harabasz_score

## EVALUATION

D_METRICS = {'internal': [silhouette_score, calinski_harabasz_score],
             'external': [adjusted_rand_score, v_measure_score, homogeneity_score, completeness_score]}

D_HYPERPARAMS = {
	'kmeans': {
			'k': np.arange(2, 10, 1)
	},
	'birch': {
			'threshold': np.linspace(0.1, 1, 10),
			'branching_factor':np.linspace(10, 100, 10, dtype=int),
			'n_clusters':np.arange(2, 10)
	},
}