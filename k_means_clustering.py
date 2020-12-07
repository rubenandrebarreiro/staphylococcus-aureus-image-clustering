# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:16:58 2020

@author: rubenandrebarreiro
"""

# Import cluster.KMeans Sub-Module,
# from SciKit-Learn Python's Library,
# as k_means
from sklearn.cluster import KMeans as k_means


from libs.visualization_and_plotting import plot_clusters_centroids_and_radii


def k_means_clustering_method(xs_features_data, num_clusters):
    
     k_means_clustering = k_means(n_clusters = num_clusters, init = 'k-means++', n_init = 10, max_iter = 300)
     
     k_means_clustering.fit(xs_features_data)
     
     ys_labels_predicted = k_means_clustering.predict(xs_features_data)
     
     clusters_centroids = k_means_clustering.cluster_centers_
     
     error_k_means_clustering = k_means_clustering.inertia_
     
     
     return ys_labels_predicted, clusters_centroids, error_k_means_clustering
     

def k_means_pre_clustering_method(xs_features_data, num_max_clusters):

    
    errors_k_means_pre_clustering = []
    
    
    for num_clusters in range( 1, ( num_max_clusters + 1 ) ):
        
        ys_labels_predicted, clusters_centroids, error_k_means_pre_clustering = k_means_clustering_method(xs_features_data, num_clusters)
        
        errors_k_means_pre_clustering.append(error_k_means_pre_clustering)
                
        plot_clusters_centroids_and_radii(xs_features_data, ys_labels_predicted, clusters_centroids, num_clusters = num_clusters, final_clustering = False)
        
        #Silhouette score
        #silhouette_score = silhouette_score(normalized, labels)
       
        #Rand index
        
        #Precision
        
        #Recall
        
        #F1 
        
        #Adjusted Rand Score
        #"rand_score = adjusted_rand_score(labels)"
        #print("Silhouette Score, for number of clusters %d is %f", i, silhouette_score)
    
    return errors_k_means_pre_clustering



def k_means_final_clustering(xs_features_data, num_clusters = 3):
    
    ys_labels_predicted, clusters_centroids, error_k_means_final_clustering = k_means_clustering_method(xs_features_data, num_clusters)
    
    plot_clusters_centroids_and_radii(xs_features_data, ys_labels_predicted, clusters_centroids, num_clusters = num_clusters, final_clustering = True)
    
    
    return error_k_means_final_clustering