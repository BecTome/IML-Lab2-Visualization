# Description: Main file for visualization of the project
# 
# To run it, use the following command:
# python main/visualization/main.py --dataset <dataset>
#
# where <dataset> is one of the following: glass, heart-h, vote

# %% [markdown]
# # Visualization on Vote Dataset

# %%
from src.read.processing import Processing
from src.decomposition.PCA import PCA
from src.clustering.KMeans import KMeans
from src.utils.utils import df_to_markdown
import config

from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, v_measure_score

from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import Isomap

from time import time
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import sys

# get arguments from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Run P2 on given dataset")

dataset = parser.parse_args().dataset

if dataset == None:
    raise Exception("Dataset not provided.")

OUTPUT_PATH = f'output/visualization/{dataset}/'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# %% [markdown]
# # 1. Load Data

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataclass = Processing(source_path='input/datasets/')
df_orig = dataclass.read(dataset).copy()
df_orig['Class'] = df_orig[config.D_TARGETS[dataset]]
dataclass.general_preprocessing()
df = dataclass.df.copy()
df['Class'] = df[config.D_TARGETS[dataset]]
if config.D_TARGETS[dataset] != 'Class':
    df.drop(columns=[config.D_TARGETS[dataset]], inplace=True)

X = df.iloc[:, :-1]
X = X.astype(float)
y = df.iloc[:, [-1]]

pca = PCA(n_components=X.shape[1])
X_transformed = pca.fit_transform(X)
df_pca = pd.DataFrame(X_transformed, columns=[f'PC{i}' for i in range(1, X.shape[1] + 1)])
df_pca['Class'] = df_orig['Class']

# %%
# Check header for our own pca
df_pca_toshow = df_pca.iloc[:5, :5].copy()
filename = os.path.join(OUTPUT_PATH, 'header_own_pca.md')
df_to_markdown(df_pca_toshow, filename)
# %%
# Statistics summary for our own pca
summ_pca_to_show = df_pca.describe().iloc[:, :5].copy()
filename = os.path.join(OUTPUT_PATH, 'summary_own_pca.md')
df_to_markdown(summ_pca_to_show, filename)


# %% [markdown]
# The classes are quite well separated in the first two components. The first component 
# is the one that separates the classes the most. The second component separates the 
# classes but not as much as the first one. It's clear that the X axis is the one with 
# higher variance and hence, the most helpful to separate the classes.

# %%
# Plot Scatter own pca
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Class', data=df_pca)
plt.title("PCA on Vote dataset (our implementation)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_own_pca.png'))
plt.close()
# %%
# Plot explained variance
plt.figure(figsize=(18, 7))
plt.plot(range(1, X.shape[1] + 1), pca.explained_variance_ratio_, linestyle='--', 
         marker='o', color='r')
plt.bar(range(1, X.shape[1] + 1), pca.explained_variance_ratio_)
for i, j in zip(range(1, X.shape[1] + 1), pca.explained_variance_ratio_):
    plt.text(i - 0.5, j + 0.01, f'{j:.2f}', fontsize=15)
plt.title("Explained variance ratio")
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio (our implementation)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'barimportance_own_pca.png'))
plt.close()
# %% [markdown]
# %%
# Same with sklearn PCA

sklearn_pca = sklearnPCA(n_components=X.shape[1])
X_sk_transformed = sklearn_pca.fit_transform(X)
df_skpca = pd.DataFrame(X_sk_transformed, 
                        columns=[f'PC{i}' for i in range(1, X.shape[1] + 1)])
df_skpca['Class'] = df_orig['Class']

df_skpca_toshow = df_skpca.iloc[:5, :5].copy()
filename = os.path.join(OUTPUT_PATH, 'header_sklearn_pca.md')
df_to_markdown(df_skpca_toshow, filename)

# %%

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Class', data=df_skpca)
plt.title("PCA on Vote dataset (sklearn implementation)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_sklearn_pca.png'))

# %%
# Plot explained variance
plt.figure(figsize=(18, 7))
plt.plot(range(1, X.shape[1] + 1), sklearn_pca.explained_variance_ratio_, 
         linestyle='--', marker='o', color='r')
plt.bar(range(1, X.shape[1] + 1), sklearn_pca.explained_variance_ratio_)
for i, j in zip(range(1, X.shape[1] + 1), sklearn_pca.explained_variance_ratio_):
    plt.text(i - 0.5, j + 0.01, f'{j:.2f}', fontsize=15)
plt.title("Explained variance ratio (sklearn implementation)")
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'barimportance_sklearn_pca.png'))

ipca = IncrementalPCA(n_components=X.shape[1])
X_ipca_transformed = ipca.fit_transform(X)
df_ipca = pd.DataFrame(X_ipca_transformed, 
                        columns=[f'PC{i}' for i in range(1, X.shape[1] + 1)])
df_ipca['Class'] = df_orig['Class']

df_ipca_toshow = df_ipca.iloc[:5, :5].copy()
filename = os.path.join(OUTPUT_PATH, 'header_ipca.md')
df_to_markdown(df_ipca_toshow, filename)

# %%
# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Class', data=df_ipca)
plt.title("Incremental PCA on Vote dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_ipca.png'))

# %%
# Plot explained variance
plt.figure(figsize=(18, 7))
plt.plot(range(1, X.shape[1] + 1), ipca.explained_variance_ratio_, linestyle='--', marker='o', color='r')
plt.bar(range(1, X.shape[1] + 1), ipca.explained_variance_ratio_)
for i, j in zip(range(1, X.shape[1] + 1), sklearn_pca.explained_variance_ratio_):
    plt.text(i - 0.5, j + 0.01, f'{j:.2f}', fontsize=15)
plt.title("Explained variance ratio (Incremental PCA)")
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'barimportance_ipca.png'))

# %% [markdown]
# The main difference is performance in time and memory. The IncrementalPCA is 
# much faster for big datasets than the PCA and it uses less memory. However, 
# the results are not exactly the same. The explained variance ratio is the same 
# but the components are not. This is because the IncrementalPCA is an approximation 
# of the PCA. The approximation is good but it is not exact.
# 
# What can be observed from the following comparison is that, when batch size is 
# increased, IPCA is faster and uses less memory. However, in this case, approximation 
# error from IPCA is negligible. This is observed from the difference in two first 
# explained variance ratio and the average difference in components. The approximation 
# error is not noticeable.
# 
# In the other hand, it doesn't either speed up the process. This is because the dataset
# is not big enough to notice the difference. However, if the dataset was bigger, the 
# difference would be noticeable.

# %%
# Compare training time for different values of batch size

batch_sizes = np.linspace(40, X.shape[0], 10, dtype=int)
times = []
explained = []
diffs = []
for batch_size in batch_sizes:
    ipca = IncrementalPCA(n_components=X.shape[1], batch_size=batch_size)
    start = time()
    ipca.fit(X)
    end = time()
    times.append(end - start)

    pct_explained_variance = ipca.explained_variance_ratio_[:2].sum()
    explained.append(pct_explained_variance)

    avg_dif = np.abs(ipca.components_ - sklearn_pca.components_).mean()
    diffs.append(avg_dif)

# Create double y axis plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(batch_sizes, times, linestyle='-', marker='o', color='r', 
         label='Training time')
ax1.set_xlabel('Batch size')
ax1.tick_params('y', colors='r')

ax2 = ax1.twinx()
ax2.plot(batch_sizes, explained, linestyle='-', marker='o', color='b', 
         label='Explained variance')
ax2.set_ylim([0, 1])

ax3 = ax1.twinx()
ax3.plot(batch_sizes, diffs, linestyle='-', marker='o', color='g', 
         label='Difference with sklearn PCA')
ax3.set_ylim([0, 1])

fig.legend(bbox_to_anchor=(0.8, 1.1), ncol=3)
plt.title("Training time and explained variance ratio of 2 first PC for different\
           batch sizes")
fig.tight_layout()

plt.savefig(os.path.join(OUTPUT_PATH, 'batch_size_comparison_time.png'))

# %% [markdown]
# For a `batch_size` equal to the number of samples, the results are the same as the PCA. This is because the algorithm is the same. The only difference is that the algorithm is implemented in a different way. The visible difference in the plots can be due to external factors such as the random initialization of the algorithm.
# 
# Our implementation of PCA is faster than the incremental one until `batch_size` of 260. However, when `batch_size` is increased, the IPCA is faster as it is expected given that sklearn PCA is faster than ours.

# %%
# Plot comparison of training time for PCA, Handmade PCA and Incremental PCA

pca = PCA(n_components=X.shape[1])
start = time()
pca.fit(X)
end = time()
pca_time = end - start

sklearn_pca = sklearnPCA(n_components=X.shape[1], random_state=0)
start = time()
sklearn_pca.fit(X)
end = time()
sklearn_pca_time = end - start

fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
ls_batch_sizes = [50, 260, 420]

for i, bs in enumerate(ls_batch_sizes):
    ipca = IncrementalPCA(n_components=X.shape[1], batch_size=bs)
    start = time()
    ipca.fit(X)
    end = time()
    ipca_time = end - start

    ax[i].bar(['PCA', 'Incremental PCA', 'Sklearn PCA'], [pca_time, ipca_time, sklearn_pca_time])
    ax[i].set_title("Training time PCA algorithms\n (batch size = {})".format(bs))
    ax[i].set_xlabel("PCA algorithm")
    ax[i].set_ylabel("Training time")

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, 'bartime_pca_algorithms.png'))

# %% [markdown]
# # 4. Use PCA with k-Means and BIRCH to compare performances

# %% [markdown]
# As it was mentioned before, after projecting the data into the 2 first Principal 
# Components, the classes are quite well separated. A lot of variables weren't 
# providing almost any information in terms of variance. This causes the well known 
# dimensionality curse, in which when the number of features is increased, the 
# performance of the algorithm decreases. This is why PCA is used, to reduce the number 
# of features and hence, the dimensionality of the data.

y = df["Class"]

n_clusters = len(np.unique(y))

birch = model_dbs = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Birch())
])

kmeans = model_dbs = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KMeans(k=n_clusters, max_iterations=100, random_state=0))])

def evaluate_model(model, X, y):

    model.fit(X)
    y_pred = model['model'].labels_
        
    try:
        silhouette = silhouette_score(X, y_pred)
    except Exception as e:
        # print(f"An error occurred during silhouette score calculation: {e}")
        silhouette = 0

    try:
        v_measure = v_measure_score(y, y_pred)
    except Exception as e:
        # print(f"An error occurred during v-measure score calculation: {e}")
        v_measure = 0

    return silhouette, v_measure

# %% [markdown]
# # Cluster the transformed Data using BIRCH

# %%
m = X.shape[1]
n_components = np.linspace(1, m, dtype=int)
birch_silhouette_scores, km_silhoette_scores = [], []
birch_v_measure_scores, km_v_measure_scores = [], []

for n in n_components:
    
    X_transformed = df_pca.iloc[:, :n]
    birch_silhouette_score, birch_v_measure_score = evaluate_model(birch, X_transformed, y)
    km_silhoette_score, km_v_measure_score = evaluate_model(kmeans, X_transformed, y)

    birch_silhouette_scores.append(birch_silhouette_score)
    birch_v_measure_scores.append(birch_v_measure_score)

    km_silhoette_scores.append(km_silhoette_score)
    km_v_measure_scores.append(km_v_measure_score)

km_silhoette_score_nored, km_v_measure_score_nored = evaluate_model(kmeans, X, y)
birch_silhoette_score_nored, birch_v_measure_score_nored = evaluate_model(birch, X, y)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
ax[0].plot(n_components, birch_silhouette_scores, linestyle='-', 
           marker='o', color='r', label='Birch')
ax[0].plot(n_components, km_silhoette_scores, linestyle='-',
              marker='o', color='b', label='KMeans')

# plot horizontal line for original values before PCA
ax[0].hlines(y=km_silhoette_score_nored, xmin=0, xmax=m, color='g', 
             linestyle='--', label='KMeans Original')

ax[0].hlines(y=birch_silhoette_score_nored, xmin=0, xmax=m, color='k', 
             linestyle='--', label='Birch Original')

ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Silhouette score')

ax[1].plot(n_components, birch_v_measure_scores, linestyle='-', 
           marker='o', color='r', label='Birch')

ax[1].plot(n_components, km_v_measure_scores, linestyle='-',
              marker='o', color='b', label='KMeans')

# plot horizontal line for original values before PCA
ax[1].hlines(y=birch_silhoette_score_nored, xmin=0, xmax=m, color='g', 
             linestyle='--', label='KMeans Original')

ax[1].hlines(y=birch_v_measure_score_nored, xmin=0, xmax=m, color='k', 
             linestyle='--', label='Birch Original')

ax[1].set_ylabel('V-measure score')
ax[1].set_xlabel('Number of components')

ax[0].legend()
ax[1].legend()

fig.suptitle("Silhouette and V-measure scores for different number of components")

fig.savefig(os.path.join(OUTPUT_PATH, 'compare_ncomp_metrics.png'))


# %% [markdown]
# As can be infered from the previous plot, in every case, incrementing the number of components even adding more information, leads to equal or worse results. This is because of the dimensionality curse. The more features, the worse the performance of the algorithm.
# 
# In the case of silhouette score, a model trained only with the first component has far better silhouette score than in the case of the model trained without applying PCA.
# 
# In the other hand, in terms of V-Measure, KMeans increases its performance when trained with just one component. However, Birch's performance slightly decreases. As it is an informed metric, it is more sensitive to the loss of information.
# 
# Another interesting fact is that both algorithms have exactly the same performance when trained with the first component.

# %%
filename_scores = "scores_pca_vs_orig.md"
# create file even if exists
open(os.path.join(OUTPUT_PATH, filename_scores), "w").close()

print("Evaluation results on BIRCH using the original dataset", 
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(birch, X, y)), 
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)

print("Evaluation results on KMeans using the original dataset", 
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(kmeans, X, y)),
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)

print("Evaluation results on Birch using the PC1", 
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(birch, df_pca.iloc[:, :1], y)),
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)

print("Evaluation results on KMeans using the PC1", 
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}".format(*evaluate_model(kmeans, df_pca.iloc[:, :1], y)),
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))


# %%
# PLot training time for different number of components
# KMeans and Birch

n_components = np.linspace(1, m, dtype=int)
birch_times, km_times = [], []

for n in n_components:
    
    X_transformed = df_pca.iloc[:, :n]

    start = time()
    birch.fit(X_transformed)
    end = time()
    birch_times.append(end - start)

    start = time()
    kmeans.fit(X_transformed)
    end = time()
    km_times.append(end - start)

# Time on non-reduced dataset
start = time()
birch.fit(X)
end = time()
birch_time_nonred = end - start

start = time()
kmeans.fit(X)
end = time()
km_time_nonred = end - start

# Plot
fig, ax = plt.subplots(1, 1, figsize=(15, 6), sharey=True)
ax.plot(n_components, birch_times, linestyle='-', 
           marker='o', color='r', label='Birch')
ax.plot(n_components, km_times, linestyle='-',
              marker='o', color='b', label='KMeans')

ax.hlines(y=birch_time_nonred, xmin=0, xmax=m, color='k',
            linestyle='--', label='Birch Original')

ax.hlines(y=km_time_nonred, xmin=0, xmax=m, color='g',
            linestyle='--', label='KMeans Original')

plt.title("Training time for different number of components")
ax.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_time_pca_kmeans_birch.png'))

# %% [markdown]
# Performance in time improves as well. This difference would be more noticeable if the dataset was bigger. From here, another advantage of dimensionality reduction method can be inferred. It is not only that the performance of the algorithm increases but also the time needed to train the algorithm decreases.

# %% [markdown]
# # 5. Cluster the transformed Data (SVD) using K-Means and Birch

# %% [markdown]
# Use sklearn.decomposition.truncatedSVD to reduce the dimensionality of your data sets
# and cluster it with your own k-Means, the one that you implemented in Work 1, and with the
# BIRCH from sklearn library. Compare your new results with the ones obtained previously. 

# %% [markdown]
# ## Non Centered Data

# %%
svd = TruncatedSVD(n_components=X.shape[1] - 1)

X_transformed_svd = svd.fit_transform(X)
df_svd = pd.DataFrame(X_transformed_svd, 
                        columns=[f'SV{i}' for i in range(1, X.shape[1])])
df_svd['Class'] = df_orig['Class']

filename = os.path.join(OUTPUT_PATH, 'header_noncentered_svd.md')
df_to_markdown(df_svd.iloc[:5, :5], filename)

# %%
# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SV1', y='SV2', hue='Class', data=df_svd)
plt.title("SVD in Vote dataset")
plt.xlabel("SV1")
plt.ylabel("SV2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_noncentered_svd.png'))

# %%
# Plot explained variance
svd_in = svd
plt.figure(figsize=(18, 7))
plt.plot(range(1, X.shape[1]), svd_in.explained_variance_ratio_, linestyle='--', marker='o', color='r')
plt.bar(range(1, X.shape[1]), svd_in.explained_variance_ratio_)
for i, j in zip(range(1, X.shape[1] + 1), svd_in.explained_variance_ratio_):
    plt.text(i - 0.5, j + 0.01, f'{j:.2f}', fontsize=15)
plt.title("Explained variance ratio (SVD)")
plt.xlabel("SV Component")
plt.ylabel("Explained variance ratio")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'barimportance_noncentered_svd.png'))

# %%
m = X.shape[1]
n_components = np.linspace(1, m - 1, dtype=int)
birch_silhouette_scores, km_silhoette_scores = [], []
birch_v_measure_scores, km_v_measure_scores = [], []

for n in n_components:

    X_transformed = df_svd.iloc[:, :n]
    birch_silhouette_score, birch_v_measure_score = evaluate_model(birch, X_transformed, y)
    km_silhoette_score, km_v_measure_score = evaluate_model(kmeans, X_transformed, y)

    birch_silhouette_scores.append(birch_silhouette_score)
    birch_v_measure_scores.append(birch_v_measure_score)

    km_silhoette_scores.append(km_silhoette_score)
    km_v_measure_scores.append(km_v_measure_score)

km_silhoette_score_nored, km_v_measure_score_nored = evaluate_model(kmeans, X, y)
birch_silhoette_score_nored, birch_v_measure_score_nored = evaluate_model(birch, X, y)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
ax[0].plot(n_components, birch_silhouette_scores, linestyle='-', 
           marker='o', color='r', label='Birch')
ax[0].plot(n_components, km_silhoette_scores, linestyle='-',
              marker='o', color='b', label='KMeans')

# plot horizontal line for original values before SVD
ax[0].hlines(y=km_silhoette_score_nored, xmin=0, xmax=m, color='g', 
             linestyle='--', label='KMeans Original')

ax[0].hlines(y=birch_silhoette_score_nored, xmin=0, xmax=m, color='k', 
             linestyle='--', label='Birch Original')

ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Silhouette score')

ax[1].plot(n_components, birch_v_measure_scores, linestyle='-', 
           marker='o', color='r', label='Birch')

ax[1].plot(n_components, km_v_measure_scores, linestyle='-',
              marker='o', color='b', label='KMeans')

# plot horizontal line for original values before SVD
ax[1].hlines(y=birch_silhoette_score_nored, xmin=0, xmax=m, color='g', 
             linestyle='--', label='KMeans Original')

ax[1].hlines(y=birch_v_measure_score_nored, xmin=0, xmax=m, color='k', 
             linestyle='--', label='Birch Original')

ax[1].set_ylabel('V-measure score')
ax[1].set_xlabel('Number of components')

ax[0].legend()
ax[1].legend()

fig.suptitle("Silhouette and V-measure scores for different number of components (not centered)")
fig.savefig(os.path.join(OUTPUT_PATH, 'compare_ncomp_metrics_noncentered.png'))

# %%
# PLot training time for different number of components

n_components = np.linspace(1, m - 1, dtype=int)
birch_times, km_times = [], []

for n in n_components:
    
    X_transformed = df_svd.iloc[:, :n]

    start = time()
    birch.fit(X_transformed)
    end = time()
    birch_times.append(end - start)

    start = time()
    kmeans.fit(X_transformed)
    end = time()
    km_times.append(end - start)

# Time on non-reduced dataset
start = time()
birch.fit(X)
end = time()
birch_time_nonred = end - start

start = time()
kmeans.fit(X)
end = time()
km_time_nonred = end - start

# Plot
fig, ax = plt.subplots(1, 1, figsize=(15, 6), sharey=True)
ax.plot(n_components, birch_times, linestyle='-', 
           marker='o', color='r', label='Birch')
ax.plot(n_components, km_times, linestyle='-',
              marker='o', color='b', label='KMeans')

ax.hlines(y=birch_time_nonred, xmin=0, xmax=m, color='k',
            linestyle='--', label='Birch Original')

ax.hlines(y=km_time_nonred, xmin=0, xmax=m, color='g',
            linestyle='--', label='KMeans Original')

plt.title("Training time for different number of components")
ax.legend()
fig.savefig(os.path.join(OUTPUT_PATH, 'training_time_noncentered_svd_kmeans_birch.png'))

# %%
filename_scores = "scores_svd_noncentered_vs_orig.md"
# create file even if exists
open(os.path.join(OUTPUT_PATH, filename_scores), "w").close()

print("Evaluation results on Birch using the original dataset", 
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(kmeans, X, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)

print("Evaluation results on KMeans using the original dataset",
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(birch, X, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)

print("Evaluation results on KMeans using the transformed dataset", 
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(kmeans, X_transformed_svd, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)
print("Evaluation results on Birch using the transformed dataset",
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(kmeans, X_transformed_svd, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
# %% [markdown]
# Behavior is strange. Singular Values are usually sorted from higher to lower magnitude. IN this case, the first one is lower than the second one and, after that, the behavior is the expected one. In Figure Scatter, there are a couple of points which are clearly separated from the rest of the distribution. These outliers are the ones that are causing this behavior.
# 
# If data gets scaled and centered, the behavior is the expected one. The first singular value is the highest one and the rest are sorted from higher to lower magnitude.
# 
# Indeed, the same behavior than PCA is observed.

# %% [markdown]
# ## Centered Version

# %%
svd = Pipeline([
                ('scaler', StandardScaler()),
                ('svd', TruncatedSVD(n_components=X.shape[1] - 1))])

X_transformed_svd = svd.fit_transform(X)
df_svd = pd.DataFrame(X_transformed_svd, 
                        columns=[f'SV{i}' for i in range(1, X.shape[1])])
df_svd['Class'] = df_orig['Class']

filename = os.path.join(OUTPUT_PATH, 'header_centered_svd.md')
df_to_markdown(df_svd.iloc[:5, :5], filename)

# %%
# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SV1', y='SV2', hue='Class', data=df_svd)
plt.title("SVD in Vote dataset")
plt.xlabel("SV1")
plt.ylabel("SV2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_centered_svd.png'))

# %%
# Plot explained variance
svd_in = svd['svd']
plt.figure(figsize=(18, 7))
plt.plot(range(1, X.shape[1]), svd_in.explained_variance_ratio_, linestyle='--', marker='o', color='r')
plt.bar(range(1, X.shape[1]), svd_in.explained_variance_ratio_)
for i, j in zip(range(1, X.shape[1] + 1), svd_in.explained_variance_ratio_):
    plt.text(i - 0.5, j + 0.01, f'{j:.2f}', fontsize=15)
plt.title("Explained variance ratio (SVD)")
plt.xlabel("SV Component")
plt.ylabel("Explained variance ratio")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'barimportance_centered_svd.png'))

# %%
m = X.shape[1]
n_components = np.linspace(1, m - 1, dtype=int)
birch_silhouette_scores, km_silhoette_scores = [], []
birch_v_measure_scores, km_v_measure_scores = [], []

for n in n_components:

    X_transformed = df_svd.iloc[:, :n]
    birch_silhouette_score, birch_v_measure_score = evaluate_model(birch, X_transformed, y)
    km_silhoette_score, km_v_measure_score = evaluate_model(kmeans, X_transformed, y)

    birch_silhouette_scores.append(birch_silhouette_score)
    birch_v_measure_scores.append(birch_v_measure_score)

    km_silhoette_scores.append(km_silhoette_score)
    km_v_measure_scores.append(km_v_measure_score)

km_silhoette_score_nored, km_v_measure_score_nored = evaluate_model(kmeans, X, y)
birch_silhoette_score_nored, birch_v_measure_score_nored = evaluate_model(birch, X, y)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
ax[0].plot(n_components, birch_silhouette_scores, linestyle='-', 
           marker='o', color='r', label='Birch')
ax[0].plot(n_components, km_silhoette_scores, linestyle='-',
              marker='o', color='b', label='KMeans')

# plot horizontal line for original values before SVD
ax[0].hlines(y=km_silhoette_score_nored, xmin=0, xmax=m, color='g', 
             linestyle='--', label='KMeans Original')

ax[0].hlines(y=birch_silhoette_score_nored, xmin=0, xmax=m, color='k', 
             linestyle='--', label='Birch Original')

ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Silhouette score')

ax[1].plot(n_components, birch_v_measure_scores, linestyle='-', 
           marker='o', color='r', label='Birch')

ax[1].plot(n_components, km_v_measure_scores, linestyle='-',
              marker='o', color='b', label='KMeans')

# plot horizontal line for original values before SVD
ax[1].hlines(y=birch_silhoette_score_nored, xmin=0, xmax=m, color='g', 
             linestyle='--', label='KMeans Original')

ax[1].hlines(y=birch_v_measure_score_nored, xmin=0, xmax=m, color='k', 
             linestyle='--', label='Birch Original')

ax[1].set_ylabel('V-measure score')
ax[1].set_xlabel('Number of components')

ax[0].legend()
ax[1].legend()

fig.suptitle("Silhouette and V-measure scores for different number of components")

fig.savefig(os.path.join(OUTPUT_PATH, 'compare_ncomp_metrics_centered.png'))



# %%
# PLot training time for different number of components

n_components = np.linspace(1, m - 1, dtype=int)
birch_times, km_times = [], []

for n in n_components:
    
    X_transformed = df_svd.iloc[:, :n]

    start = time()
    birch.fit(X_transformed)
    end = time()
    birch_times.append(end - start)

    start = time()
    kmeans.fit(X_transformed)
    end = time()
    km_times.append(end - start)

# Time on non-reduced dataset
start = time()
birch.fit(X)
end = time()
birch_time_nonred = end - start

start = time()
kmeans.fit(X)
end = time()
km_time_nonred = end - start

# Plot
fig, ax = plt.subplots(1, 1, figsize=(15, 6), sharey=True)
ax.plot(n_components, birch_times, linestyle='-', 
           marker='o', color='r', label='Birch')
ax.plot(n_components, km_times, linestyle='-',
              marker='o', color='b', label='KMeans')

ax.hlines(y=birch_time_nonred, xmin=0, xmax=m, color='k',
            linestyle='--', label='Birch Original')

ax.hlines(y=km_time_nonred, xmin=0, xmax=m, color='g',
            linestyle='--', label='KMeans Original')

plt.title("Training time for different number of components")
ax.legend()
plt.savefig(os.path.join(OUTPUT_PATH, 'training_time_centered_svd_kmeans_birch.png'))

# %% [markdown]
# The results are exactly the same than for the case of PCA. 

# %%
filename_scores = "scores_svd_centered_vs_orig.md"
# create file even if exists
open(os.path.join(OUTPUT_PATH, filename_scores), "w").close()

print("Evaluation results on BIRCH using the transformed dataset", 
      file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(birch, X_transformed_svd, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)

print("Evaluation results on KMeans using the transformed dataset",
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(kmeans, X_transformed_svd, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)

print("Evaluation results on BIRCH using the original dataset",
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(birch, X, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))

print("-"*50)
print("Evaluation results on KMeans using the original dataset",
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))
print("Silhouette {:.2f} - V-Measure {:.2f}\n".format(*evaluate_model(kmeans, X, y)),
        file=open(os.path.join(OUTPUT_PATH, filename_scores), "a"))


# %%
# Visualize the original Dataset
plt.figure(figsize=(10, 6))
plt.scatter(df.to_numpy()[:, 0], df.to_numpy()[:, 1], c=df["Class"])
plt.title("First 2 dimensions from Original Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_original.png'))

birch.fit(X)
kmeans.fit(X)

# Visualize the results of BIRCH and KMenas on the original Dataset
figure, axs = plt.subplots(1, 2, figsize=(15, 6))
axs[0].scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 1], c=birch['model'].labels_)
axs[0].set_title("BIRCH results on the original Dataset")
axs[0].set_xlabel("X1")
axs[0].set_ylabel("X2")

axs[1].scatter(X.to_numpy()[:, 0], X.to_numpy()[:, 1], c=kmeans['model'].labels_)
axs[1].set_title("KMeans results on the original Dataset")
axs[1].set_xlabel("X1")
axs[1].set_ylabel("X2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_original_birch_kmeans.png'))
matplotlib.pyplot.close()
# %%
isomap = Isomap(n_components=2)

X_transformed = pca.fit_transform(X)
X_transformed_isomap = isomap.fit_transform(X)

# Visualize results of BIRCH and KMeans on the original Dataset using PCA and ISOMAP
figure, axs = plt.subplots(2, 2, figsize=(15, 12))
axs[0, 0].scatter(X_transformed[:, 0], X_transformed[:, 1], c=birch['model'].labels_)
axs[0, 0].set_title("BIRCH on the original Dataset using PCA")
axs[0, 0].set_xlabel("PC1")
axs[0, 0].set_ylabel("PC2")

axs[0, 1].scatter(X_transformed[:, 0], X_transformed[:, 1], c=kmeans['model'].labels_)
axs[0, 1].set_title("KMeans on the original Dataset using PCA")
axs[0, 1].set_xlabel("PC1")
axs[0, 1].set_ylabel("PC2")

axs[1, 0].scatter(X_transformed_isomap[:, 0], X_transformed_isomap[:, 1], c=birch['model'].labels_)
axs[1, 0].set_title("BIRCH on the original Dataset using ISOMAP")
axs[1, 0].set_xlabel("PC1")
axs[1, 0].set_ylabel("PC2")

axs[1, 1].scatter(X_transformed_isomap[:, 0], X_transformed_isomap[:, 1], c=kmeans['model'].labels_)
axs[1, 1].set_title("KMeans on the original Dataset using ISOMAP")
axs[1, 1].set_xlabel("PC1")
axs[1, 1].set_ylabel("PC2")
plt.savefig(os.path.join(OUTPUT_PATH, 'scatter_birch_kmeans_pca_isomap.png'))


