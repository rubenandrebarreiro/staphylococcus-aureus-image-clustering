# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:35:35 2020

@author: Martim Figueiredo

@author: rubenandrebarreiro
"""

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import arange,
# From the NumPy's Python Library,
# as a_range
from numpy import arange as a_range

# Import NanMax,
# From the NumPy's Python Library,
# as nan_max
from numpy import nanmax as nan_max

# Import arange,
# From the NumPy's Python Library,
# as array_matrix
from numpy import array as array_matrix

# Import Cluster.DBSCAN Sub-Module,
# from SciKit-Learn Python's Library,
# as dbscan
from sklearn.cluster import DBSCAN as dbscan


from libs.visualization_and_plotting import plot_clusters_centroids_and_radii

from libs.visualization_and_plotting import plot_silhouette_analysis

from libs.visualization_and_plotting import plot_confusion_matrix_rand_index_clustering_heatmap

from libs.visualization_and_plotting import plot_clustering_scores


from libs.performance_scoring_metrics import compute_clustering_performance_metrics

from libs.performance_scoring_metrics import print_dbscan_clustering_performance_metrics



def dbscan_clustering_method(xs_features_data, current_epsilon, num_closest_k_neighbors = 5):
    
    dbscan_clustering = dbscan(eps = current_epsilon, min_samples = num_closest_k_neighbors)
    
    dbscan_clustering.fit(xs_features_data)
    
    ys_labels_predicted = dbscan_clustering.labels_
    
    
    clusters_centroids_indices = dbscan_clustering.core_sample_indices_
    

    xs_features_data_inliers = xs_features_data[ys_labels_predicted != -1]
    
    xs_features_data_outliers = xs_features_data[ys_labels_predicted == -1]

    
    clusters_centroids_points = xs_features_data[clusters_centroids_indices, :]

    clusters_border_points = array_matrix([list(point) for point in xs_features_data_inliers if point not in clusters_centroids_points])
    
    
    return ys_labels_predicted, clusters_centroids_indices, clusters_centroids_points, clusters_border_points, xs_features_data_inliers, xs_features_data_outliers


def dbscan_pre_clustering_method(xs_features_data, ys_labels_true, num_closest_k_neighbors = 5, start_epsilon = 0.01, end_epsilon = 0.28, step_epsilon = 0.01):
    
    current_epsilon_step = 0
    
    num_epsilons_steps = int( ( end_epsilon - start_epsilon ) / step_epsilon )
    
    clusters_epsilon_values = matrix_array_zeros( num_epsilons_steps )
    clusters_num_centroids = matrix_array_zeros( num_epsilons_steps )
    
    clusters_num_inliers = matrix_array_zeros( num_epsilons_steps )
    clusters_num_outliers = matrix_array_zeros( num_epsilons_steps )
    
    clusters_silhouette_scores = matrix_array_zeros( num_epsilons_steps )
    clusters_precision_scores = matrix_array_zeros( num_epsilons_steps )
    clusters_recall_scores = matrix_array_zeros( num_epsilons_steps )
    clusters_rand_index_scores = matrix_array_zeros( num_epsilons_steps )
    clusters_f1_scores = matrix_array_zeros( num_epsilons_steps )
    clusters_adjusted_rand_scores = matrix_array_zeros( num_epsilons_steps )
    
    
    for current_epsilon in a_range(start_epsilon, end_epsilon, step_epsilon):
       
        ys_labels_predicted, clusters_centroids_indices, clusters_centroids_points, clusters_border_points, xs_features_data_inliers, xs_features_data_outliers = dbscan_clustering_method(xs_features_data, current_epsilon, num_closest_k_neighbors = num_closest_k_neighbors)

        num_clusters_centroids = ( nan_max(ys_labels_predicted) + 1 )
        
        plot_clusters_centroids_and_radii("DBScan", xs_features_data, ys_labels_predicted, clusters_centroids_points, num_clusters = num_clusters_centroids, epsilon = current_epsilon, final_clustering = False)
        
        
        clusters_num_centroids[current_epsilon_step] = num_clusters_centroids
        clusters_num_inliers[current_epsilon_step] = len(xs_features_data_inliers)
        clusters_num_outliers[current_epsilon_step] = len(xs_features_data_outliers)
        
        
        if(num_clusters_centroids >= 2):
            
            plot_silhouette_analysis("DBScan", xs_features_data, ys_labels_predicted, clusters_centroids_points, num_clusters_centroids, epsilon = current_epsilon, final_clustering = False)
        
            silhouette_score, precision_score, recall_score, rand_index_score, f1_score, adjusted_rand_score, confusion_matrix_rand_index_clustering = compute_clustering_performance_metrics("K-Means", xs_features_data, ys_labels_true, ys_labels_predicted, num_clusters_centroids, final_clustering = False)
            
            plot_confusion_matrix_rand_index_clustering_heatmap("DBScan", confusion_matrix_rand_index_clustering, num_clusters_centroids, epsilon = current_epsilon, final_clustering = False)
            
            
            clusters_silhouette_scores[current_epsilon_step] = silhouette_score
            clusters_precision_scores[current_epsilon_step] = precision_score
            clusters_recall_scores[current_epsilon_step] = recall_score
            clusters_rand_index_scores[current_epsilon_step] = rand_index_score
            clusters_f1_scores[current_epsilon_step] = f1_score
            clusters_adjusted_rand_scores[current_epsilon_step] = adjusted_rand_score
            
        else:
            
            clusters_silhouette_scores[current_epsilon_step] = -1.0
            clusters_precision_scores[current_epsilon_step] = -1.0
            clusters_recall_scores[current_epsilon_step] = -1.0
            clusters_rand_index_scores[current_epsilon_step] = -1.0
            clusters_f1_scores[current_epsilon_step] = -1.0
            clusters_adjusted_rand_scores[current_epsilon_step] = -1.0
            
        current_epsilon_step = ( current_epsilon_step + 1 )
    
    
    print_dbscan_clustering_performance_metrics("DBScan", start_epsilon, end_epsilon, step_epsilon, clusters_num_centroids, clusters_num_inliers, clusters_num_outliers, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    plot_clustering_scores("DBScan", 0, start_epsilon, end_epsilon, step_epsilon, clusters_epsilon_values, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    
    return clusters_num_centroids, clusters_num_inliers, clusters_num_outliers, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores

