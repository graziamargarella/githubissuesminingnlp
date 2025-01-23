import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.cluster import KMeans

PATH_DATA = './results/feature_engineering'
PATH_RESULTS = './results/clustering'
EPS_VALUES = np.arange(0.1, 1, 0.05)
MIN_SAMPLE_VALUES = np.arange(5,40,5)
N_CLUSTERS_VALUES = np.arange(2, 35, 1)
MODELS_FILE_NAMES = ["db_best.pkl", "best_spect.pkl", "db_best_commit.pkl", "best_spect_commit.pkl"]
PLOTS_FILE_NAMES = ['clustering_DBSCAN.pdf', 'clustering_spectral.pdf', 'clustering_DBSCAN_commits.pdf', 'clustering_spectral_commits.pdf']


def data_loading():
    """
    Function to load the data extracted by the feature engineering process.
    :return df: the dataframe scaled and as identifier the login of the developer
    :return df_commit: the dataframe of the only commit data scaled and as identifier the login of the developer 
    """
    scaler = StandardScaler().set_output(transform='pandas')

    df = pd.read_csv(os.path.join(PATH_DATA, 'all_features.csv')).set_index('author')
    df_commit = pd.read_csv(os.path.join(PATH_DATA, 'user_features_commits.csv')).set_index('author').dropna()

    df = scaler.fit_transform(df)
    df_commit = scaler.fit_transform(df_commit)
    return df, df_commit


def grid_search_dbscan(X, eps_values, min_samples_values, file_name):
    """
    Function to execute the grid search of the two parameters of the DBSCAN clustering algorithm,
        and save in pickle format the instance of the best parameters obtained comparing the silhouette score.
    :param X: the dataset to evaluate.
    :param eps_values: a list of values for the eps parameter.
    :param min_samples_values: a list of values for the min_samples_values.
    :param file_name: name of the pickle file in which will be saved the best model.
    :return best_params: a dictionary with the best configuration obtained.
    :return best_score: the silhouette score obtained by the best configuration.
    :return best_db: the instance of the best algorithm configuration.
    """
    best_score = -1
    best_params = None
    best_db = None
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
            labels = db.fit_predict(X)
            
            if len(set(labels)) > 1: # Computes the silhouette score if there are at least 2 clusters
                score = silhouette_score(X, labels, metric='manhattan')
                print(f"DBSCAN -> eps: {eps}, min_samples: {min_samples}, silhouette_score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_db = db
                    with open(os.path.join(PATH_RESULTS, file_name), "wb") as f:
                        try:
                            pickle.dump(best_db, f)
                        except InconsistentVersionWarning as w:
                            pass
    
    return best_params, best_score, best_db


def grid_search_spectral(X, n_clusters_values, file_name):
    """
    Function to execute the grid search of the two parameters of the SpectralClustering clustering algorithm,
        and save in pickle format the instance of the best parameters obtained comparing the silhouette score.
    :param X: the dataset to evaluate.
    :param n_clusters_values: a list of values for the n_clusters parameter.
    :param file_name: name of the pickle file in which will be saved the best model.
    :return best_params: a dictionary with the best configuration obtained.
    :return best_score: the silhouette score obtained by the best configuration.
    :return best_db: the instance of the best algorithm configuration.
    """
    best_score = -1
    best_params = None
    best_spect = None
    for n_clusters in n_clusters_values:
        sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='discretize', random_state=42)
        labels = sc.fit_predict(X)

        if len(set(labels)) > 1: # Computes the silhouette score if there are at least 2 clusters
            score = silhouette_score(X, labels, metric='manhattan')
            print(f"SpectralClustering -> n_clusters: {n_clusters}, silhouette_score: {score}")

            if score > best_score:
                best_score = score
                best_params = {'n_clusters': n_clusters}
                best_spect = sc
                with open(os.path.join(PATH_RESULTS, file_name), "wb") as f:
                    try:
                        pickle.dump(best_spect, f)
                    except InconsistentVersionWarning as w:
                        pass
                   
    return best_params, best_score, best_spect


def best_model_loading(file_name):
    with open(os.path.join(PATH_RESULTS, file_name), "rb") as f:
        model = pickle.load(f)
    return model

def visualizations(X, labels, file_name_plot1, cluster_means, file_name_plot2):
    """
    Function to generate the plot about the clustering visualization with their labels,
        and to generate the heatmap of each feature grouped by clusters.
    :param X: datapoint after the t-SNE procedure (2 dimensions).
    :param labels: labels of the datapoints in the X set.
    :param file_name_plot1: string of the file name to save the results of the first plot.
    :param cluster_means: dataframe with the mean value of each feature per each cluster.
    :param file_name_plot2: string of the file name to save the results of the second plot.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=labels,
            colorscale='Viridis', 
            showscale=True,
            opacity=0.8
        )
    ))

    fig.update_layout(legend=dict(font=dict(size=20),itemwidth=30), plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.update_xaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside', showline=True)
    fig.update_yaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside', showline=True)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.write_image(os.path.join(PATH_RESULTS, file_name_plot1))
    #fig.show() 

    plt.figure(figsize=(30, 20))
    sns.heatmap(cluster_means.T, cmap="coolwarm", annot=True, fmt=".2f", vmax=3)
    sns.set_theme(font_scale=20)
    plt.xlabel('Cluster')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_RESULTS, file_name_plot2), bbox_inches = 'tight')
    #plt.show()
        
def dbscan_experiments(mode='load'):
    """
    Function to execute the gridsearch on DBSCAN parameters or execute the best one, already searched.
    :param mode: string to choose if 'load' or 'search' the best model.
    """
    df, df_commit = data_loading()

    if mode == 'load':
        db_best = best_model_loading('db_best.pkl')
        db_best_commit = best_model_loading('db_best_commit.pkl')
    elif mode == 'search':
        best_params_dbscan, best_score_dbscan, db_best = grid_search_dbscan(df, EPS_VALUES, MIN_SAMPLE_VALUES,'db_best.pkl')
        print(f"Best DBSCAN params: {best_params_dbscan}, Best silhouette score: {best_score_dbscan}")
        best_params_dbscan_commit, best_score_dbscan_commit, db_best_commit = grid_search_dbscan(df_commit, EPS_VALUES, MIN_SAMPLE_VALUES, 'db_best_commit.pkl')
        print(f"Best DBSCAN params: {best_params_dbscan_commit}, Best silhouette score: {best_score_dbscan_commit}")
    elif mode != 'load' or mode != 'search':
        return -1
    
    labels = db_best.labels_
    tsne = TSNE(n_components=2, metric='manhattan')
    X = tsne.fit_transform(df)
    df['cluster'] = labels
    cluster_means = df.groupby('cluster').mean()
    visualizations(X, labels, 'clustering_DBSCAN.pdf', cluster_means, 'distribution_all_dbscan.pdf')
    
    labels_commit = db_best_commit.labels_
    tsne_commit = TSNE(n_components=2, metric='manhattan')
    X_commit = tsne_commit.fit_transform(df_commit)
    df_commit['cluster'] = labels_commit
    cluster_means = df_commit.groupby('cluster').mean()
    visualizations(X_commit, labels_commit, 'clustering_commit_DBSCAN.pdf', cluster_means, 'distribution_commit_dbscan.pdf')


def spectral_experiments(mode='load'):
    """
    Function to execute the gridsearch on SpectralClustering parameters or execute the best one, already searched.
    :param mode: string to choose if 'load' or 'search' the best model.
    """
    df, df_commit = data_loading()

    if mode == 'load':
        best_spect = best_model_loading('best_spect.pkl')
        best_spect_commit = best_model_loading('best_spect_commit.pkl')
    elif mode == 'search':
        best_params_spectral, best_score_spectral,best_spect = grid_search_spectral(df, N_CLUSTERS_VALUES, 'best_spect.pkl')
        print(f"Best Spectral Clustering params: {best_params_spectral}, Best silhouette score: {best_score_spectral}")
        best_params_spectral_commit, best_score_spectral_commit, best_spect_commit = grid_search_spectral(df_commit, N_CLUSTERS_VALUES, 'best_spect_commit.pkl')
        print(f"Best Spectral Clustering params: {best_params_spectral_commit}, Best silhouette score: {best_score_spectral_commit}")
    elif mode != 'load' or mode != 'search':
        return -1
    
    labels = best_spect.labels_
    tsne = TSNE(n_components=2, metric='manhattan')
    X = tsne.fit_transform(df)
    df['cluster'] = labels
    cluster_means = df.groupby('cluster').mean()
    visualizations(X, labels, 'clustering_spectral.pdf', cluster_means, 'distribution_all_spectral.pdf')
    

    labels_commit = best_spect_commit.labels_
    tsne_commit = TSNE(n_components=2, metric='manhattan')
    X_commit = tsne_commit.fit_transform(df_commit)
    df_commit['cluster'] = labels_commit
    cluster_means = df_commit.groupby('cluster').mean()
    visualizations(X_commit, labels_commit, 'clustering_commit_spectral.pdf', cluster_means, 'distribution_commit_spectral.pdf')
    