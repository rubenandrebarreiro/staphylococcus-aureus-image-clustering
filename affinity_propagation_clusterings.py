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


from libs.visualization_and_plotting import plot_clusters_centroids_and_radii

from libs.visualization_and_plotting import plot_silhouette_analysis

from libs.visualization_and_plotting import plot_confusion_matrix_rand_index_clustering_heatmap

from libs.visualization_and_plotting import plot_clustering_scores


from libs.performance_scoring_metrics import compute_clustering_performance_metrics

from libs.performance_scoring_metrics import print_dbscan_clustering_performance_metrics



def affinity_propagation_clustering_method(xs_features_data, damping_value = 0.5, max_iterations = 300):
    
     affinity_propagation_clustering = affinity_propagation(damping = damping_value, preference=-50)
     
     affinity_propagation_clustering.fit(xs_features_data)
     
     ys_labels_predicted = affinity_propagation_clustering.predict(xs_features_data)
     
     clusters_centroids_indices = affinity_propagation_clustering.cluster_centers_indices_
     
     clusters_centroids_points = xs_features_data[clusters_centroids_indices, :]
     
     
     return ys_labels_predicted, clusters_centroids_indices, clusters_centroids_points
 

def affinity_propagation_pre_clustering(xs_features_data, ys_labels_true, start_damping = 0.01, end_damping = 0.28, step_damping = 0.01):
    
    current_damping_step = 0
    
    num_damping_steps = int( ( end_damping - start_damping ) / step_damping )
    
    clusters_epsilon_values = matrix_array_zeros( num_damping_steps )
    clusters_num_centroids = matrix_array_zeros( num_damping_steps )
    
    clusters_silhouette_scores = matrix_array_zeros( num_damping_steps )
    clusters_precision_scores = matrix_array_zeros( num_damping_steps )
    clusters_recall_scores = matrix_array_zeros( num_damping_steps )
    clusters_rand_index_scores = matrix_array_zeros( num_damping_steps )
    clusters_f1_scores = matrix_array_zeros( num_damping_steps )
    clusters_adjusted_rand_scores = matrix_array_zeros( num_damping_steps )
    
    
    for current_damping in a_range(start_damping, end_damping, step_damping):
        
        ys_labels_predicted, clusters_centroids_indices, clusters_centroids_points = affinity_propagation_clustering_method(xs_features_data, damping_value = current_damping, max_iterations = 300)
                
        num_clusters_centroids = ( nan_max(ys_labels_predicted) + 1 )
        
        plot_clusters_centroids_and_radii("Affinity-Propagation", xs_features_data, ys_labels_predicted, clusters_centroids_points, num_clusters = num_clusters_centroids, epsilon = None, final_clustering = False)
        
        clusters_num_centroids[current_damping_step] = num_clusters_centroids
        
        
        if(num_clusters_centroids >= 2):
            
            plot_silhouette_analysis("Affinity-Propagation", xs_features_data, ys_labels_predicted, clusters_centroids_points, num_clusters_centroids, epsilon = None, final_clustering = False)
        
            silhouette_score, precision_score, recall_score, rand_index_score, f1_score, adjusted_rand_score, confusion_matrix_rand_index_clustering = compute_clustering_performance_metrics("Affinity-Propagation", xs_features_data, ys_labels_true, ys_labels_predicted, num_clusters_centroids, final_clustering = False)
            
            plot_confusion_matrix_rand_index_clustering_heatmap("Affinity-Propagation", confusion_matrix_rand_index_clustering, num_clusters_centroids, epsilon = None, final_clustering = False)
                        
            clusters_silhouette_scores[current_damping_step] = silhouette_score
            clusters_precision_scores[current_damping_step] = precision_score
            clusters_recall_scores[current_damping_step] = recall_score
            clusters_rand_index_scores[current_damping_step] = rand_index_score
            clusters_f1_scores[current_damping_step] = f1_score
            clusters_adjusted_rand_scores[current_damping_step] = adjusted_rand_score
    
        else:
            
            clusters_silhouette_scores[current_damping_step] = -1.0
            clusters_precision_scores[current_damping_step] = -1.0
            clusters_recall_scores[current_damping_step] = -1.0
            clusters_rand_index_scores[current_damping_step] = -1.0
            clusters_f1_scores[current_damping_step] = -1.0
            clusters_adjusted_rand_scores[current_damping_step] = -1.0
            
        current_damping_step = ( current_damping_step + 1 )
    
    print_dbscan_clustering_performance_metrics("Affinity-Propagation", start_damping, end_damping, step_damping, clusters_num_centroids, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    plot_clustering_scores("Affinity-Propagation", 0, start_damping, end_damping, step_damping, clusters_epsilon_values, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores)
    
    
    return clusters_num_centroids, clusters_silhouette_scores, clusters_precision_scores, clusters_recall_scores, clusters_rand_index_scores, clusters_f1_scores, clusters_adjusted_rand_scores
