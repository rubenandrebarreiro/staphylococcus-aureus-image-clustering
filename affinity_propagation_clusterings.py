# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:25:17 2020

@author: rubenandrebarreiro
"""

# Import zeros,
# From the NumPy's Python Library,
# as matrix_array_zeros
from numpy import zeros as matrix_array_zeros

# Import arange,
# From the NumPy's Python Library,
# as array_matrix
from numpy import array as array_matrix

# Import arange,
# From the NumPy's Python Library,
# as a_range
from numpy import arange as a_range

# Import NanMax,
# From the NumPy's Python Library,
# as nan_max
from numpy import nanmax as nan_max

# Import sort,
# From the NumPy's Python Library,
# as array_sort
from numpy import sort as array_sort

# Import unique,
# From the NumPy's Python Library,
# as array_unique
from numpy import unique as array_unique

from sklearn.cluster import AffinityPropagation as affinity_propagation


def affinity_propagation_clustering_method(xs_features_data, damping_value = 0.5, max_iterations = 300):
    
     affinity_propagation_clustering = affinity_propagation(damping = damping_value, preference=-50)
     
     affinity_propagation_clustering.fit(xs_features_data)
     
     ys_labels_predicted = affinity_propagation_clustering.predict(xs_features_data)
     
     clusters_centroids_indices = affinity_propagation_clustering.cluster_centers_indices_
     
     clusters_centroids_points = xs_features_data[clusters_centroids_indices, :]
     
     
     return ys_labels_predicted, clusters_centroids_indices, clusters_centroids_points
 

def affinity_propagation_pre_clustering(xs_features_data, ys_labels_true, start_damping = 0.01, end_damping = 0.28, step_damping = 0.01):
    
    current_epsilon_step = 0
    
    num_damping_steps = int( ( end_damping - start_damping ) / step_damping )
    
    clusters_silhouette_scores = matrix_array_zeros( num_damping_steps )
    clusters_precision_scores = matrix_array_zeros( num_damping_steps )
    clusters_recall_scores = matrix_array_zeros( num_damping_steps )
    clusters_rand_index_scores = matrix_array_zeros( num_damping_steps )
    clusters_f1_scores = matrix_array_zeros( num_damping_steps )
    clusters_adjusted_rand_scores = matrix_array_zeros( num_damping_steps )
    
    
    for current_damping in a_range(start_damping, end_damping, step_damping):
        
        ys_labels_predicted, cluster_centers_indices, cluster_labels = affinity_propagation_clustering_method(xs_features_data, damping_value = current_damping, max_iterations = 300)
                
        
        plot_clusters_centroids_and_radii("Affinity-Propagation", xs_features_data, ys_labels_predicted, clusters_centroids, num_clusters = current_num_clusters, epsilon = None, final_clustering = False)
        
        
        if(current_num_clusters >= 2):
            
            plot_silhouette_analysis("Affinity-Propagation", xs_features_data, ys_labels_predicted, clusters_centroids, current_num_clusters, epsilon = None, final_clustering = False)
        
            silhouette_score, precision_score, recall_score, rand_index_score, f1_score, adjusted_rand_score, confusion_matrix_rand_index_clustering = compute_clustering_performance_metrics("K-Means", xs_features_data, ys_labels_true, ys_labels_predicted, current_num_clusters, final_clustering = False)
            
            plot_confusion_matrix_rand_index_clustering_heatmap("Affinity-Propagation", confusion_matrix_rand_index_clustering, current_num_clusters, epsilon = None, final_clustering = False)
                        
            clusters_silhouette_scores[( current_num_clusters - 2 )] = silhouette_score
            clusters_precision_scores[( current_num_clusters - 2 )] = precision_score
            clusters_recall_scores[( current_num_clusters - 2 )] = recall_score
            clusters_rand_index_scores[( current_num_clusters - 2 )] = rand_index_score
            clusters_f1_scores[( current_num_clusters - 2 )] = f1_score
            clusters_adjusted_rand_scores[( current_num_clusters - 2 )] = adjusted_rand_score
    
    
    print_k_means_clustering_performance_metrics("Affinity-Propagation", num_total_clusters, clusters_squared_errors_sums_intertias, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    plot_clustering_scores("Affinity-Propagation", num_total_clusters, 0, 0, 0, None, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    
    return clusters_squared_errors_sums_intertias, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores
