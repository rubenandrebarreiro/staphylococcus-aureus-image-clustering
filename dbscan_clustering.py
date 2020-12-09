# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:35:35 2020

@author: Martim Figueiredo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:16:58 2020

@author: rubenandrebarreiro
"""

# Import cluster.DBSCAN Sub-Module,
# from SciKit-Learn Python's Library,
# as dbscan

from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame as pandas_data_frame

from sklearn.preprocessing import MinMaxScaler

from libs.visualization_and_plotting import plot_clusters_centroids_and_radii_dbscan


def dbscan_clustering_method(xs_features_data, epsilon):
    
    norm_data = MinMaxScaler()
    X = norm_data.fit_transform(xs_features_data)
    
    dbscan_clustering = DBSCAN(eps = epsilon, min_samples = 5)
     
    dbscan_clustering.fit(X)
     
    ys_labels_predicted = dbscan_clustering.labels_
    
    clusters_centroids = dbscan_clustering.core_sample_indices_
    
    outliers = X[ys_labels_predicted == -1]
    centroids = X[ys_labels_predicted != -1]

    print(ys_labels_predicted)
    print("\n")
    print(clusters_centroids)
    
    
    #FALTA ORDERNAR OS VALORES COMO DIZ NO ENUNCIADO
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    
    # plot
    
    plt.figure(figsize=(20,10))
    plt.plot(distances)
    plt.title('K-distance Graph',fontsize=20)
    plt.xlabel('Data Points sorted by distance',fontsize=14)
    plt.ylabel('Epsilon',fontsize=14)
    plt.savefig( 'plots/{}-with-{}-dbscan' + str(epsilon) + '-{}.png')
    
    #plot_clusters_centroids_and_radii_dbscan(xs_features_data, ys_labels_predicted, centroids, num_clusters = len(clusters_centroids), final_clustering = False)
    plt.scatter(xs_features_data[ys_labels_predicted == 0, 0], xs_features_data[ys_labels_predicted == 0, 1], color = 'b')
    plt.scatter(xs_features_data[ys_labels_predicted == 1, 0], xs_features_data[ys_labels_predicted == 1, 1], color = 'r')
    plt.scatter(xs_features_data[ys_labels_predicted == 2, 0], xs_features_data[ys_labels_predicted == 2, 1], color = 'g')
    plt.scatter(xs_features_data[ys_labels_predicted == 3, 0], xs_features_data[ys_labels_predicted == 3, 1], color = 'y')
    # Plot the Centroids of the Clusters, as Scatter Points
    plt.scatter(xs_features_data[ys_labels_predicted == -1, 0], xs_features_data[ys_labels_predicted == -1, 1], color = 'b')
        
    # Set the Title of the K-Means Clustering, for K Clusters
    plt.title( 'DBSCAN Clustering, with K = {} Cluster(s)'.format(epsilon) )     
        
    # Save the Plot, as a figure/image
    plt.savefig( 'imgs/dbscan-clustering-for-{}-clusters-centroids.png'.format(epsilon))
    
  
    return ys_labels_predicted, clusters_centroids
    
     

def dbscan_pre_clustering_method(xs_features_data, epsilon_max):
     
    errors_dbscan_pre_clustering = []
    epsilon = 0.01
    
    for epsilon in np.arange(0.01, 0.3, 0.01):
       
        ys_labels_predicted, clusters_centroids = dbscan_clustering_method(xs_features_data, epsilon)
     
       
    return errors_dbscan_pre_clustering
    
    
   